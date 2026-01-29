#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

download() {
  url="$1"
  out="$2"
  if [ -f "$out" ]; then
    echo "exists: $out"
    return
  fi
  echo "downloading $out"
  curl -L "$url" -o "$out"
}

download "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" "$MODELS_DIR/face_detection_yunet_2023mar.onnx"
download "https://media.githubusercontent.com/media/opencv/opencv_zoo/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" "$MODELS_DIR/face_recognition_sface_2021dec.onnx"

echo "models ready in $MODELS_DIR"
