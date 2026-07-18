use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{bail, Context, Result};

/// Module index of the created null-sink, 0 = none. Global so the panic hook
/// can tear it down without threading state through unwinding.
static MODULE_IDX: AtomicU32 = AtomicU32::new(0);

pub struct VirtMic {
    pub node_name: String,
}

impl VirtMic {
    /// Unload stale modules from a previous SIGKILL'd run (matched by exact
    /// sink_name argument - never touches other virtual devices).
    pub fn cleanup_stale(name: &str) {
        if let Ok(out) = Command::new("pactl").args(["list", "short", "modules"]).output() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                if line.contains("module-null-sink") && line.contains(&format!("sink_name={name}")) {
                    if let Some(idx) = line.split_whitespace().next() {
                        let _ = Command::new("pactl").args(["unload-module", idx]).status();
                    }
                }
            }
        }
    }

    /// Create a virtual source "VC-Mic" apps see as a normal microphone.
    pub fn create(name: &str) -> Result<Self> {
        Self::cleanup_stale(name);
        let out = Command::new("pactl")
            .args([
                "load-module",
                "module-null-sink",
                &format!("sink_name={name}"),
                &format!("sink_properties=device.description={name}"),
                "media.class=Audio/Source/Virtual",
                "audio.position=[MONO]",
            ])
            .output()
            .context("pactl not found - pipewire-pulse required")?;
        if !out.status.success() {
            bail!("pactl load-module failed: {}", String::from_utf8_lossy(&out.stderr));
        }
        let idx: u32 = String::from_utf8_lossy(&out.stdout).trim().parse().context("module index parse")?;
        MODULE_IDX.store(idx, Ordering::SeqCst);
        Ok(Self { node_name: name.to_string() })
    }
}

pub fn teardown() {
    let idx = MODULE_IDX.swap(0, Ordering::SeqCst);
    if idx != 0 {
        let _ = Command::new("pactl").args(["unload-module", &idx.to_string()]).status();
    }
}

impl Drop for VirtMic {
    fn drop(&mut self) {
        teardown();
    }
}

/// Install teardown on panic + SIGINT/SIGTERM. Call once at startup.
pub fn install_guards() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        teardown();
        default_hook(info);
        // a dead worker with a live main loop = zombie daemon; die loudly
        std::process::exit(101);
    }));
    let _ = ctrlc::set_handler(|| {
        teardown();
        std::process::exit(130);
    });
}

/// Query pw-dump for a node's object.serial by exact node.name.
pub fn node_serial(name: &str) -> Option<String> {
    let out = Command::new("pw-dump").output().ok()?;
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    for o in v.as_array()? {
        let props = match o.get("info").and_then(|i| i.get("props")) {
            Some(p) => p,
            None => continue,
        };
        if props.get("node.name").and_then(|n| n.as_str()) == Some(name) {
            if let Some(serial) = props.get("object.serial") {
                return Some(serial.to_string());
            }
        }
    }
    None
}

/// Query pw-dump for a node's numeric id by exact node.name.
pub fn node_id(name: &str) -> Option<u32> {
    let out = Command::new("pw-dump").output().ok()?;
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    for o in v.as_array()? {
        let props = match o.get("info").and_then(|i| i.get("props")) {
            Some(p) => p,
            None => continue,
        };
        if props.get("node.name").and_then(|n| n.as_str()) == Some(name) {
            return o.get("id").and_then(|i| i.as_u64()).map(|i| i as u32);
        }
    }
    None
}

/// Pick the first real hardware capture (alsa_input.*) - avoids virtual
/// sources like the python client's VC_Mic being the system default.
pub fn pick_hw_source() -> Option<String> {
    let out = Command::new("pw-dump").output().ok()?;
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    for o in v.as_array()? {
        let props = match o.get("info").and_then(|i| i.get("props")) {
            Some(p) => p,
            None => continue,
        };
        let class = props.get("media.class").and_then(|c| c.as_str()).unwrap_or("");
        let name = props.get("node.name").and_then(|c| c.as_str()).unwrap_or("");
        if class == "Audio/Source" && name.starts_with("alsa_input.") {
            return Some(name.to_string());
        }
    }
    None
}
