"""fp16-convert an onnx model (shape-infer first). Run: ... in.onnx out.onnx"""
import sys

import onnx
from onnx import shape_inference
from onnxconverter_common import float16

m = onnx.load(sys.argv[1])
try:
    m = shape_inference.infer_shapes(m)
except Exception as e:  # >2GB models can fail proto-level inference
    print("shape_infer skipped:", e, file=sys.stderr)
m16 = float16.convert_float_to_float16(m, keep_io_types=True)
onnx.save(m16, sys.argv[2])
print("ok", sys.argv[2])

