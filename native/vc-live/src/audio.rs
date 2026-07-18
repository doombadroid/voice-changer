use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use pipewire as pw;
use pw::{properties::properties, spa};
use rtrb::{Consumer, Producer};
use spa::pod::Pod;

pub static CAPTURE_XRUNS: AtomicU64 = AtomicU64::new(0);
pub static PLAYBACK_UNDERRUNS: AtomicU64 = AtomicU64::new(0);

fn format_param(rate: u32) -> Vec<u8> {
    let mut info = spa::param::audio::AudioInfoRaw::new();
    info.set_format(spa::param::audio::AudioFormat::F32LE);
    info.set_channels(1);
    info.set_rate(rate);
    let obj = spa::pod::Object {
        type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: spa::param::ParamType::EnumFormat.as_raw(),
        properties: info.into(),
    };
    spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(obj),
    )
    .unwrap()
    .0
    .into_inner()
}

/// Capture stream: default (or targeted) mic -> 16 kHz mono f32 ring.
/// Returned StreamBox must stay alive; runs on the caller's mainloop.
pub fn capture_stream<'a>(
    core: &'a pw::core::CoreRc,
    target: Option<&str>,
    mut tx: Producer<f32>,
) -> Result<pw::stream::StreamBox<'a>> {
    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Communication",
        *pw::keys::NODE_NAME => "vc-live-capture",
    };
    if let Some(t) = target {
        props.insert(*pw::keys::TARGET_OBJECT, t);
    }
    let stream = pw::stream::StreamBox::new(core, "vc-live-capture", props)?;
    let _listener = stream
        .add_local_listener_with_user_data(())
        .param_changed(|_, _, id, param| {
            if id == spa::param::ParamType::Format.as_raw() {
                if let Some(param) = param {
                    let mut info = spa::param::audio::AudioInfoRaw::new();
                    if info.parse(param).is_ok() {
                        eprintln!("CAPTURE FORMAT: rate {} ch {}", info.rate(), info.channels());
                    }
                }
            }
        })
        .process(move |stream, _| {
            if let Some(mut buffer) = stream.dequeue_buffer() {
                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    return;
                }
                let d = &mut datas[0];
                let n_bytes = d.chunk().size() as usize;
                if let Some(slice) = d.data() {
                    let n = n_bytes / 4;
                    let mut dropped = false;
                    for i in 0..n {
                        let v = f32::from_le_bytes(slice[i * 4..i * 4 + 4].try_into().unwrap());
                        if tx.push(v).is_err() {
                            dropped = true;
                        }
                    }
                    if dropped {
                        CAPTURE_XRUNS.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        })
        .register()?;
    // listener must outlive the stream: leak it alongside (held by caller via Box)
    std::mem::forget(_listener);
    let values = format_param(16000);
    let mut params = [Pod::from_bytes(&values).unwrap()];
    stream.connect(
        spa::utils::Direction::Input,
        None,
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut params,
    )?;
    Ok(stream)
}

/// Playback stream: output ring (40 kHz mono f32) -> target node (virtual mic).
pub fn playback_stream<'a>(
    core: &'a pw::core::CoreRc,
    target: &str,
    target_id: Option<u32>,
    out_rate: u32,
    mut rx: Consumer<f32>,
    warm: Arc<std::sync::atomic::AtomicBool>,
) -> Result<pw::stream::StreamBox<'a>> {
    let props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Playback",
        *pw::keys::MEDIA_ROLE => "Communication",
        *pw::keys::NODE_NAME => "vc-live-out",
        *pw::keys::TARGET_OBJECT => target,
        "node.dont-reconnect" => "true",
    };
    let stream = pw::stream::StreamBox::new(core, "vc-live-out", props)?;
    let _listener = stream
        .add_local_listener_with_user_data(())
        .param_changed(|_, _, id, param| {
            if id == spa::param::ParamType::Format.as_raw() {
                if let Some(param) = param {
                    let mut info = spa::param::audio::AudioInfoRaw::new();
                    if info.parse(param).is_ok() {
                        eprintln!("PLAYBACK FORMAT: rate {} ch {}", info.rate(), info.channels());
                    }
                }
            }
        })
        .process(move |stream, _| {
            if let Some(mut buffer) = stream.dequeue_buffer() {
                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    return;
                }
                let d = &mut datas[0];
                let n_frames = if let Some(slice) = d.data() {
                    let n = slice.len() / 4;
                    let ready = warm.load(Ordering::Relaxed);
                    let mut wrote = 0usize;
                    for i in 0..n {
                        let v = if ready { rx.pop().unwrap_or(0.0) } else { 0.0 };
                        slice[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                        wrote += 1;
                    }
                    if ready && rx.is_empty() && wrote > 0 {
                        PLAYBACK_UNDERRUNS.fetch_add(1, Ordering::Relaxed);
                    }
                    n
                } else {
                    0
                };
                let chunk = d.chunk_mut();
                *chunk.offset_mut() = 0;
                *chunk.stride_mut() = 4;
                *chunk.size_mut() = (n_frames * 4) as _;
            }
        })
        .register()?;
    std::mem::forget(_listener);
    let values = format_param(out_rate);
    let mut params = [Pod::from_bytes(&values).unwrap()];
    // NO AUTOCONNECT: WirePlumber refuses to route playback streams into
    // Audio/Source/Virtual nodes (tried name/serial/node-id targets); we link
    // ports manually instead (see link_playback_to_mic).
    let _ = target_id;
    let _ = target;
    stream.connect(
        spa::utils::Direction::Output,
        None,
        pw::stream::StreamFlags::MAP_BUFFERS | pw::stream::StreamFlags::RT_PROCESS,
        &mut params,
    )?;
    Ok(stream)
}

/// Manually link vc-live-out:output_MONO -> <mic>:input_FL/FR, retrying until
/// the ports exist (runs in a background thread).
pub fn link_playback_to_mic(mic_name: String) {
    std::thread::spawn(move || {
        for _ in 0..50 {
            let mut ok = false;
            for port in ["input_FL", "input_FR"] {
                let st = std::process::Command::new("pw-link")
                    .args(["vc-live-out:output_MONO", &format!("{mic_name}:{port}")])
                    .output();
                if let Ok(o) = st {
                    let err = String::from_utf8_lossy(&o.stderr);
                    if o.status.success() || err.contains("File exists") {
                        ok = true;
                    }
                }
            }
            if ok {
                eprintln!("playback linked to {mic_name}");
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        eprintln!("WARNING: could not link playback to {mic_name}");
    });
}
