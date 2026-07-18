fn main() {
    // Codegen order: smallest first so survivors still build if one chokes.
    // Comment out a .input(...) line to skip a failing model (record the error).
    burn_onnx::ModelGen::new()
        .input("../golden/fcpe_fp32.onnx")
        .input("../golden/synth_alexjones_static49.onnx")
        .input("../golden/rmvpe_20231006.onnx")
        .input("../../server/pretrain/content_vec_500.onnx")
        .out_dir("model/")
        .run_from_script();
}
