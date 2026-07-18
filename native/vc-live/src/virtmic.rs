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
    }));
    let _ = ctrlc::set_handler(|| {
        teardown();
        std::process::exit(130);
    });
}
