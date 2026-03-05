# howrs

**Howrs** is a high-performance facial recognition authentication system written in Rust, designed as a modern alternative to Howdy. It provides fast and secure face-based authentication for Linux systems, with support for PAM integration.

## Features

- **Secure**: Uses state-of-the-art YuNet face detection and SFace recognition models
- **PAM Integration**: Drop-in replacement for password authentication
- **Hardware Acceleration**: Supports multiple device(onnx), including NPU/GPU/CPU

## Get Started

- Fedora: Install `howrs` from copr(https://copr.fedorainfracloud.org/coprs/eason0729/howrs/)
- Binary: Download from release.
The binary is single ELF, which is both a CLI(for face enrollment) and a cdylib(for PAM).

It store data in following location:
- config file: `/usr/local/etc/howrs/config.toml`
- face embeddings: `/usr/local/etc/howrs`

> [!TIP]
> They are readonly for normal user, so enrollment require root

## Usage

```
howrs --help
Howrs - modern facial recognition authentication

Usage: howrs <COMMAND>

Commands:
  enroll  Enroll face from camera
  test    Test authentication by matching against enrolled faces
  purge   Remove all enrolled faces for a user
  config  Open config file in editor
  help    Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

### PAM Configuration

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

### Configuration

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

### Troubleshooting

1. Choosing Camera

```bash
# Test the camera
ffplay /dev/video0
```

2. SELinux

We have SELinux policy at `packaging`, rpm package should have that handled.

## Build from source

### System Dependencies

- Rust nightly toolchain (automatically configured via `rust-toolchain.toml`)
- V4L2 compatible camera
- ONNX Runtime (automatically downloaded during build)

### Steps

1. Download Models

Before building, you must download the ONNX models:

```bash
bash howrs-vision/models/download_models.sh
```

This downloads:
- `face_detection_yunet_2023mar.onnx` - YuNet face detector
- `face_recognition_sface_2021dec.onnx` - SFace recognition model

2. compile the binary

> [!IMPORTANT]
> On x86_64, we recommend following `RUSTFLAGS`:
> ```bash
> export RUSTFLAGS="-C target-cpu=x86-64-v2 -C target-feature=+avx2"
> ```

```bash
cargo build --bin howrs --release
```

### Build with Specific Execution Provider

> [!TIP]
> All builds will fallback to CPU if runtime library isn't installed

```bash
# For NVIDIA GPU acceleration
cargo build --bin howrs --release --features cuda

# For Intel OpenVINO (default)
cargo build --bin howrs --release --features openvino
```

### Optional Build Environment Variables

- `HOWRS_CONFIG_PATH` - Path to config file (default: `/usr/local/etc/howrs/config.toml`)
- `HOWRS_FACE_STORE_PREFIX` - Path for face data storage (default: `/usr/local/etc/howrs`)

These are embedded at compile time via `option_env!`:

```bash
HOWRS_CONFIG_PATH=/etc/howrs/config.toml HOWRS_FACE_STORE_PREFIX=/etc/howrs cargo build --bin howrs --release
```

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
