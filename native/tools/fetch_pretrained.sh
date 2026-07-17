#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
URL="https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx"
OUT="native/golden/rmvpe_20231006.onnx"
if ! { [ -s "$OUT" ] && [ "$(stat -c%s "$OUT")" -gt 100000000 ]; }; then
  curl -L --fail -o "$OUT" "$URL"
fi
stat -c '%n %s' "$OUT"
sha256sum "$OUT"
# repair the 110-byte AccessDenied stub in the python install's pretrain dir
if [ "$(stat -c%s server/pretrain/rmvpe.onnx 2>/dev/null || echo 0)" -lt 1000 ]; then
  cp "$OUT" server/pretrain/rmvpe.onnx && echo "repaired server/pretrain/rmvpe.onnx"
fi
