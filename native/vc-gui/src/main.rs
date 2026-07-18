use std::process::Command;
use std::sync::atomic::Ordering;

use eframe::egui;
use serde::{Deserialize, Serialize};
use vc_live::livecore::{self, LiveConfig, LiveHandle};

#[derive(Serialize, Deserialize, Clone)]
struct Settings {
    v: u32,
    input: Option<String>,
    block: usize,
    ctx_left: usize,
    crossfade: usize,
    lookahead: usize,
    pitch: i32,
    seed: u64,
    monitor: bool,
}

impl Default for Settings {
    fn default() -> Self {
        let c = LiveConfig::default();
        Self {
            v: 1,
            input: None,
            block: c.block,
            ctx_left: c.ctx_left,
            crossfade: c.crossfade,
            lookahead: c.lookahead,
            pitch: 0,
            seed: c.seed,
            monitor: false,
        }
    }
}

fn settings_path() -> std::path::PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".config"));
    base.join("vc-native/settings.json")
}

fn load_settings() -> Settings {
    std::fs::read(settings_path())
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

fn save_settings(s: &Settings) {
    let p = settings_path();
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(b) = serde_json::to_vec_pretty(s) {
        let _ = std::fs::write(p, b);
    }
}

fn list_sources() -> Vec<(String, String)> {
    let mut out = Vec::new();
    if let Ok(o) = Command::new("pw-dump").output() {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&o.stdout) {
            if let Some(arr) = v.as_array() {
                for obj in arr {
                    let props = match obj.get("info").and_then(|i| i.get("props")) {
                        Some(p) => p,
                        None => continue,
                    };
                    let class = props.get("media.class").and_then(|c| c.as_str()).unwrap_or("");
                    if class != "Audio/Source" {
                        continue;
                    }
                    let name = props.get("node.name").and_then(|c| c.as_str()).unwrap_or("");
                    if name.is_empty() || name == "vc_mic" {
                        continue;
                    }
                    let desc = props
                        .get("node.description")
                        .and_then(|c| c.as_str())
                        .unwrap_or(name)
                        .to_string();
                    out.push((name.to_string(), desc));
                }
            }
        }
    }
    out
}

struct App {
    s: Settings,
    dirty_saved: bool,
    handle: Option<LiveHandle>,
    devices: Vec<(String, String)>,
    monitor_module: Option<u32>,
    last_error: Option<String>,
    in_peak: f32,
    out_peak: f32,
}

impl App {
    fn new() -> Self {
        Self {
            s: load_settings(),
            dirty_saved: true,
            handle: None,
            devices: list_sources(),
            monitor_module: None,
            last_error: None,
            in_peak: 0.0,
            out_peak: 0.0,
        }
    }

    fn cfg(&self) -> LiveConfig {
        LiveConfig {
            input: self.s.input.clone(),
            block: self.s.block,
            ctx_left: self.s.ctx_left,
            crossfade: self.s.crossfade,
            lookahead: self.s.lookahead,
            pitch: self.s.pitch,
            seed: self.s.seed,
            ..Default::default()
        }
    }

    fn start(&mut self) {
        match livecore::start(self.cfg()) {
            Ok(h) => {
                self.last_error = None;
                self.handle = Some(h);
                if self.s.monitor {
                    self.set_monitor(true);
                }
            }
            Err(e) => self.last_error = Some(format!("{e:#}")),
        }
    }

    fn stop(&mut self) {
        self.set_monitor(false);
        if let Some(h) = self.handle.take() {
            h.stop();
        }
    }

    fn restart(&mut self) {
        self.stop();
        self.start();
    }

    fn set_monitor(&mut self, on: bool) {
        if on && self.monitor_module.is_none() {
            if let Ok(o) = Command::new("pactl")
                .args(["load-module", "module-loopback", "source=vc_mic", "latency_msec=60"])
                .output()
            {
                if o.status.success() {
                    self.monitor_module = String::from_utf8_lossy(&o.stdout).trim().parse().ok();
                }
            }
        } else if !on {
            if let Some(idx) = self.monitor_module.take() {
                let _ = Command::new("pactl").args(["unload-module", &idx.to_string()]).status();
            }
        }
    }
}

fn ms(samples: usize) -> f64 {
    samples as f64 / 16.0
}

fn vu(ui: &mut egui::Ui, label: &str, peak: f32) {
    let db = 20.0 * (peak.max(1e-5)).log10();
    let frac = ((db + 60.0) / 60.0).clamp(0.0, 1.0);
    ui.horizontal(|ui| {
        ui.label(label);
        let color = if frac > 0.9 {
            egui::Color32::RED
        } else if frac > 0.7 {
            egui::Color32::YELLOW
        } else {
            egui::Color32::from_rgb(80, 200, 120)
        };
        let bar = egui::ProgressBar::new(frac).desired_width(220.0).fill(color);
        ui.add(bar);
        ui.monospace(format!("{db:5.1} dB"));
    });
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let running = self.handle.is_some();

        if let Some(h) = &self.handle {
            let inr = h.stats.in_rms_milli.load(Ordering::Relaxed) as f32 / 1000.0;
            let outr = h.stats.out_rms_milli.load(Ordering::Relaxed) as f32 / 1000.0;
            self.in_peak = self.in_peak.max(inr) * 0.92 + inr * 0.08;
            self.out_peak = self.out_peak.max(outr) * 0.92 + outr * 0.08;
        } else {
            self.in_peak *= 0.9;
            self.out_peak *= 0.9;
        }

        egui::Frame::central_panel(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                let color = if running {
                    egui::Color32::from_rgb(80, 200, 120)
                } else {
                    egui::Color32::GRAY
                };
                ui.colored_label(color, "●");
                ui.heading("vc-native");
                ui.separator();
                if running {
                    if ui.button("⏹ Stop").clicked() {
                        self.stop();
                    }
                } else if ui.button("▶ Start").clicked() {
                    self.start();
                }
                let mut mute = self
                    .handle
                    .as_ref()
                    .map(|h| h.stats.mute.load(Ordering::Relaxed))
                    .unwrap_or(false);
                if ui
                    .toggle_value(&mut mute, "🔇 MUTE")
                    .on_hover_text("panic mute - output silence")
                    .changed()
                {
                    if let Some(h) = &self.handle {
                        h.stats.mute.store(mute, Ordering::Relaxed);
                    }
                }
                let mut mon = self.s.monitor;
                if ui
                    .toggle_value(&mut mon, "🎧 Monitor")
                    .on_hover_text("hear the converted voice on your speakers")
                    .changed()
                {
                    self.s.monitor = mon;
                    self.dirty_saved = false;
                    if running {
                        self.set_monitor(mon);
                    }
                }
                if let Some(h) = &self.handle {
                    ui.separator();
                    ui.label(format!("in: {}", h.stats.capture_node.lock().unwrap()));
                }
            });
            if let Some(e) = &self.last_error {
                ui.colored_label(egui::Color32::RED, e);
            }
            ui.separator();
            vu(ui, "mic ", self.in_peak);
            vu(ui, "out ", self.out_peak);
            ui.separator();

            let mut changed = false;

            ui.horizontal(|ui| {
                ui.label("Pitch");
                if ui
                    .add(egui::Slider::new(&mut self.s.pitch, -12..=12).suffix(" st"))
                    .changed()
                {
                    changed = true;
                    if let Some(h) = &self.handle {
                        h.stats.pitch.store(self.s.pitch, Ordering::Relaxed);
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Input");
                let current = self.s.input.clone().unwrap_or_else(|| "(auto: usb mic)".to_string());
                egui::ComboBox::from_id_salt("input")
                    .width(340.0)
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(self.s.input.is_none(), "(auto: usb mic)").clicked() {
                            self.s.input = None;
                            changed = true;
                        }
                        for (name, desc) in &self.devices {
                            let sel = self.s.input.as_deref() == Some(name);
                            if ui.selectable_label(sel, format!("{desc}  [{name}]")).clicked() {
                                self.s.input = Some(name.clone());
                                changed = true;
                            }
                        }
                    });
                if ui.button("⟳").on_hover_text("refresh device list").clicked() {
                    self.devices = list_sources();
                }
            });

            ui.separator();
            ui.label("Engine window (Apply restarts the stream):");
            fn param_slider(
                ui: &mut egui::Ui,
                label: &str,
                v: &mut usize,
                lo: usize,
                hi: usize,
                changed: &mut bool,
            ) {
                ui.horizontal(|ui| {
                    ui.label(label);
                    let mut steps = (*v / 320) as i64;
                    let r = ui.add(
                        egui::Slider::new(&mut steps, (lo / 320) as i64..=(hi / 320) as i64)
                            .custom_formatter(|s, _| format!("{:.0} ms", s * 20.0))
                            .custom_parser(|t| t.trim_end_matches(" ms").trim().parse::<f64>().ok().map(|m| m / 20.0)),
                    );
                    if r.changed() {
                        *v = steps as usize * 320;
                        *changed = true;
                    }
                });
            }
            param_slider(ui, "Block     ", &mut self.s.block, 1280, 8000, &mut changed);
            param_slider(ui, "Context   ", &mut self.s.ctx_left, 1280, 16000, &mut changed);
            param_slider(ui, "Crossfade ", &mut self.s.crossfade, 320, 1920, &mut changed);
            param_slider(ui, "Lookahead ", &mut self.s.lookahead, 0, 4800, &mut changed);

            ui.horizontal(|ui| {
                if ui.add_enabled(running, egui::Button::new("Apply (restart stream)")).clicked() {
                    self.restart();
                }
                let hop = self
                    .handle
                    .as_ref()
                    .map(|h| h.stats.hop_p50_us.load(Ordering::Relaxed) as f64 / 1000.0)
                    .unwrap_or(ms(self.s.block));
                ui.label(format!(
                    "est. mouth-to-ear ≈ {:.0} ms",
                    ms(self.s.block) / 2.0 + hop + ms(self.s.crossfade + 320 + self.s.lookahead) + 40.0
                ));
            });

            ui.separator();
            if let Some(h) = &self.handle {
                let st = &h.stats;
                let p50 = st.hop_p50_us.load(Ordering::Relaxed) as f64 / 1000.0;
                let p95 = st.hop_p95_us.load(Ordering::Relaxed) as f64 / 1000.0;
                let budget = ms(h.cfg.block);
                ui.label(format!(
                    "hops {}   misses {}   internal latency {} ms",
                    st.hops.load(Ordering::Relaxed),
                    st.misses.load(Ordering::Relaxed),
                    st.internal_latency_ms.load(Ordering::Relaxed),
                ));
                ui.horizontal(|ui| {
                    ui.label("hop");
                    let frac = ((p50 / budget) as f32 / 1.5).clamp(0.0, 1.0);
                    let color = if p50 > budget {
                        egui::Color32::RED
                    } else if p95 > budget {
                        egui::Color32::YELLOW
                    } else {
                        egui::Color32::from_rgb(80, 200, 120)
                    };
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .desired_width(260.0)
                            .fill(color)
                            .text(format!("p50 {p50:.0} / p95 {p95:.0} / budget {budget:.0} ms")),
                    );
                });
            } else {
                ui.label("stopped");
            }

            if changed {
                self.dirty_saved = false;
            }
        });

        if !self.dirty_saved {
            save_settings(&self.s);
            self.dirty_saved = true;
        }
        if running {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        }
    }

    fn on_exit(&mut self) {
        self.stop();
    }
}

fn main() -> eframe::Result<()> {
    pipewire::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([560.0, 500.0])
            .with_title("vc-native"),
        ..Default::default()
    };
    eframe::run_native(
        "vc-native",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_theme(egui::Theme::Dark);
            Ok(Box::new(App::new()))
        }),
    )
}
