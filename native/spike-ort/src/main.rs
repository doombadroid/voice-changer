use anyhow::Result;

fn main() -> Result<()> {
    // Probe build: confirms ort links + which EPs compiled in. Real CLI lands with T6.
    println!("ort build info: {}", ort::info());
    Ok(())
}
