use anyhow::Result;
use clap::Parser;

mod virtmic;

#[derive(Parser)]
#[command(about = "vc-native live voice changer (M2)")]
struct Args {
    /// enumerate pipewire audio nodes and exit
    #[arg(long, default_value_t = false)]
    probe: bool,
    /// create the virtual mic, hold until Ctrl-C (lifecycle test)
    #[arg(long, default_value_t = false)]
    hold_mic: bool,
    /// virtual mic node name
    #[arg(long, default_value = "vc_mic")]
    mic_name: String,
}

fn probe() -> Result<()> {
    let mainloop = pipewire::main_loop::MainLoopRc::new(None)?;
    let context = pipewire::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let registry = core.get_registry()?;

    let ml = mainloop.clone();
    // stop the loop once the initial registry dump has flushed (sync trick)
    let pending = core.sync(0)?;
    let _sync_listener = core
        .add_listener_local()
        .done(move |id, seq| {
            if id == pipewire::core::PW_ID_CORE && seq == pending {
                ml.quit();
            }
        })
        .register();

    let _reg_listener = registry
        .add_listener_local()
        .global(|global| {
            if let Some(props) = &global.props {
                if let Some(class) = props.get("media.class") {
                    if class.starts_with("Audio/") {
                        println!(
                            "{:5} {:24} {}",
                            global.id,
                            class,
                            props.get("node.description").or(props.get("node.name")).unwrap_or("?")
                        );
                    }
                }
            }
        })
        .register();

    mainloop.run();
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    pipewire::init();
    if args.probe {
        probe()?;
    }
    if args.hold_mic {
        virtmic::install_guards();
        let _mic = virtmic::VirtMic::create(&args.mic_name)?;
        eprintln!("virtual mic '{}' up; Ctrl-C to exit", args.mic_name);
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
        }
    }
    unsafe { pipewire::deinit() };
    Ok(())
}
