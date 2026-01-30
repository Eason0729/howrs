# howrs

**Howrs** is a high-performance facial recognition authentication system written in Rust, designed as a modern alternative to Howdy. It provides fast and secure face-based authentication for Linux systems, with support for PAM integration.

## Features

- **Secure**: Uses state-of-the-art YuNet face detection and SFace recognition models
- **PAM Integration**: Drop-in replacement for password authentication
- **Hardware Acceleration**: Supports multiple device(onnx), including NPU/GPU/CPU
- **Blazing Fast**: Optimized with SIMD instructions for post-processing

## Architecture

Howrs consists of two main components:

- **howrs** - Main binary and PAM module for authentication
- **howrs-vision** - Face detection and recognition library with ONNX Runtime backend

The system uses:
- **YuNet** for fast and accurate face detection
- **SFace** for generating discriminative face embeddings
- **SIMD optimizations** for dot product calculations and image processing
- **Postcard** for compact, version-stable binary serialization

## Requirements

### System Dependencies

- Rust nightly toolchain (automatically configured via `rust-toolchain.toml`)
- V4L2 compatible camera
- ONNX Runtime (automatically downloaded during build)

### Optional Hardware Acceleration

The following execution providers can be enabled via Cargo features:
- `cuda` - NVIDIA GPU acceleration
- `openvino` - Intel CPU/GPU optimization (default)
- `coreml` - Apple Silicon acceleration
- `tensorrt` - NVIDIA TensorRT
- `directml` - DirectX Machine Learning
- And many more (see `Cargo.toml` for full list)

## Compilation

### Standard Build

```bash
# Clone the repository
git clone <repository-url>
cd howrs

# Build in release mode (recommended)
cargo build --release

# The binaries will be in target/release/
```

### Build with Specific Execution Provider

> [!TIP]
> All build will fallback to CPU if runtime library isn't installed

```bash
# For NVIDIA GPU acceleration
cargo build --release --features cuda

# For Intel OpenVINO (default)
cargo build --release --features openvino
```

## Installation

```bash
# Build the project
cargo build --release

# Install the binary
sudo install -m 755 target/release/howrs /usr/local/bin/

# Install the PAM module
sudo install -m 644 target/release/libhowrs.so /usr/local/lib/security/pam_howrs.so

# Create configuration directory
sudo mkdir -p /usr/local/etc/howrs

# Create default configuration
cat <<EOF | sudo tee /usr/local/etc/howrs/config.toml
threshold = 0.6
camera = "/dev/video0"
scan_durnation = 5
EOF
```

## Usage

### Enroll Your Face

```bash
# Enroll current user
howrs enroll

# Enroll specific user (requires sudo)
sudo howrs enroll --user username
```

The enrollment process will:
1. Open the configured camera
2. Capture up to 30 frames
3. Detect and select the best quality face
4. Store the face embedding in `/usr/local/etc/howrs/<username>/faces.bin`

### Test Authentication

```bash
# Test authentication for current user
howrs test

# Test for specific user
sudo howrs test --user username
```

### Remove Enrolled Faces

```bash
# Remove all faces for current user
howrs purge

# Remove for specific user
sudo howrs purge --user username
```

## PAM Configuration

### Basic Setup

To enable facial recognition authentication, edit your PAM configuration files.

**For sudo authentication** (`/etc/pam.d/sudo`):

```
# Add this line at the top, before other auth lines
auth sufficient pam_howrs.so

# Existing auth lines
auth include system-auth
```

**For login screen** (`/etc/pam.d/system-auth` or `/etc/pam.d/common-auth`):

```
# Add before other auth methods
auth sufficient pam_howrs.so

# Existing auth lines
auth required pam_unix.so
```

## Configuration

### Main Configuration File

Located at `/usr/local/etc/howrs/config.toml`, can be open with `howrs config`:

```toml
# Similarity threshold for authentication (0.0 - 1.0)
# Higher = stricter matching
# Recommended: 0.6 - 0.8
threshold = 0.6

# Camera device path
camera = "/dev/video0"

# How long the scan take
scan_durnation = 5
```

## Troubleshooting

### Choosing Camera

```bash
# Test the camera
ffplay /dev/video0
```

### Low Recognition Accuracy

- Ensure good lighting conditions
- Position face directly facing camera
- Adjust `threshold` value in config (lower = more lenient)
- Enroll multiple times from different angles

### PAM Module Not Working

```bash
# Check PAM module is installed
ls -l /usr/local/lib/security/pam_howrs.so

# Check PAM configuration syntax
sudo pam-auth-update --force

# View PAM logs
sudo journalctl -xe | grep pam_howrs
```

### Illegal Instruction

The distributed package target x86 feature level v2 and AVX2, so you might need to build your own package.

## Security Considerations

1. **Not a Sole Authentication Method** - Always configure as `sufficient` in PAM, not `required`, to allow password fallback
2. **Physical Access** - Face authentication is vulnerable to photographs/videos (consider liveness detection in future)
3. **Storage Security** - Face embeddings are stored in `/usr/local/etc/howrs/`, owned by root
4. **Privacy** - Raw images are never stored, only mathematical embeddings
5. **Threshold Tuning** - Balance security vs convenience by adjusting the similarity threshold

## Acknowledgments

- Inspired by [Howdy](https://github.com/boltgolt/howdy)
- Uses [YuNet](https://github.com/ShiqiYu/libfacedetection) for face detection
- Uses [SFace](https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface) for face recognition
- Built with [ONNX Runtime](https://onnxruntime.ai/)
