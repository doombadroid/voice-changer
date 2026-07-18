pub mod audio;
pub mod livecore;
pub mod virtmic;

use anyhow::Result;

pub fn probe() -> Result<()> {
    let mainloop = pipewire::main_loop::MainLoopRc::new(None)?;
    let context = pipewire::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let registry = core.get_registry()?;

    let ml = mainloop.clone();
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
