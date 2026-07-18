fn main() {
    // Codegen order: smallest first so survivors still build if one chokes.
    // Comment out a .input(...) line to skip a failing model (record the error).
    burn_onnx::ModelGen::new()
        .input("../golden/fcpe_fp32.onnx")
        .input("../golden/synth_alexjones_static49.onnx")
        // rmvpe skipped: runtime pads unsupported by burn-onnx 0.21; re-exportable
        // static from rmvpe.pt if quality-mode-on-GPU is ever needed.
        .input("../../server/pretrain/content_vec_500.onnx")
        .out_dir("model/")
        .run_from_script();

    // burn-onnx 0.21 emits reverse-Slice (steps=-1, end=i64::MIN+1, from the
    // flow's torch.flip) as an out-of-range i32 literal in ndarray's s![] —
    // rewrite to Tensor::flip until fixed upstream.
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("model");
    for entry in std::fs::read_dir(&out).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            let s = std::fs::read_to_string(&p).unwrap();
            let fixed = s.replace(
                ".slice(s![.., - 1.. - 9223372036854775807; - 1, ..])",
                ".flip([1])",
            );
            if fixed != s {
                std::fs::write(&p, fixed).unwrap();
            }
        }
    }
}
