Name:           howrs
Version:        0.1.0
Release:        1%{?dist}
Summary:        High-performance facial recognition authentication system for Linux
License:        Apache-2.0
URL:            https://github.com/Eason0729/howrs/
Source0:        https://github.com/Eason0729/howrs/

BuildRequires:  rust >= 1.70
BuildRequires:  cargo
BuildRequires:  clang-devel
BuildRequires:  systemd-rpm-macros

Requires:       pam
Requires:       libv4l

%description
Howrs is a high-performance facial recognition authentication system written in
Rust, designed as a modern alternative to Howdy. It provides fast and secure
face-based authentication for Linux systems with PAM integration.

Features:
- Secure: Uses state-of-the-art YuNet face detection and SFace recognition models
- PAM Integration: Drop-in replacement for password authentication
- Hardware Acceleration: Supports multiple device(onnx), including NPU/GPU/CPU
- Blazing Fast: Optimized with SIMD instructions for post-processing

This package includes pre-downloaded ONNX models for offline building.

%prep
%autosetup

# Verify ONNX models are included in the source tarball
if [ ! -f howrs-vision/models/face_detection_yunet_2023mar.onnx ] || \
   [ ! -f howrs-vision/models/face_recognition_sface_2021dec.onnx ]; then
    echo "Error: ONNX models not found in source tarball"
    echo "Please run packaging/build-srpm-fedpkg.sh to generate a proper source tarball"
    exit 1
fi

%build

# Only use x86-64-v2 and AVX2 on x86_64 architecture
%ifarch x86_64
export RUSTFLAGS="-C target-cpu=x86-64-v2 -C target-feature=+avx2"
%endif

# Compile CLI
cargo build --bin --release --features openvino

# Compile PAM module
cargo build --lib --release --features openvino

%install
# Install binary
install -D -m 755 target/release/howrs %{buildroot}%{_sbindir}/howrs

# Install PAM module
install -D -m 755 target/release/libhowrs.so %{buildroot}/%{_lib}/security/libhowrs.so

# Install default configuration
install -D -m 644 packaging/config.toml %{buildroot}/usr/local/etc/howrs/config.toml

# Create face storage directory (readable by all users, but only root can write)
install -d -m 755 %{buildroot}/usr/local/etc/howrs

# Install SELinux policy (if it exists and is not empty)
if [ -s packaging/howrs_pam.pp ]; then
    install -D -m 644 packaging/howrs_pam.pp %{buildroot}%{_datadir}/selinux/packages/%{name}/howrs_pam.pp
fi

%post
# Load SELinux policy module (if available)
if [ $1 -eq 1 ] ; then
    if [ -f %{_datadir}/selinux/packages/%{name}/howrs_pam.pp ]; then
        if command -v semodule > /dev/null 2>&1 && selinuxenabled 2>/dev/null; then
            semodule -i %{_datadir}/selinux/packages/%{name}/howrs_pam.pp 2>/dev/null || :
            restorecon -R %{_lib}/security/libhowrs.so 2>/dev/null || :
            restorecon -R %{_sbindir}/howrs 2>/dev/null || :
            restorecon -R /usr/local/etc/howrs 2>/dev/null || :
        fi
    fi
fi

# Set permissions for face data directory
if [ -d /usr/local/etc/howrs ]; then
    chmod 755 /usr/local/etc/howrs
    find /usr/local/etc/howrs -type d -exec chmod 755 {} \; 2>/dev/null || :
    find /usr/local/etc/howrs -type f -exec chmod 644 {} \; 2>/dev/null || :
fi

%postun
# Remove SELinux policy module on uninstall (if it was installed)
if [ $1 -eq 0 ] ; then
    if command -v semodule > /dev/null 2>&1 && selinuxenabled 2>/dev/null; then
        if semodule -l 2>/dev/null | grep -q howrs_pam; then
            semodule -r howrs_pam 2>/dev/null || :
        fi
    fi
fi

%files
%license LICENSE
%doc README.md
%{_sbindir}/howrs
/%{_lib}/security/libhowrs.so
%dir /usr/local/etc/howrs
%config(noreplace) /usr/local/etc/howrs/config.toml
%dir %{_datadir}/selinux/packages/%{name}
%{_datadir}/selinux/packages/%{name}/howrs_pam.pp

%changelog
* Thu Jan 30 2025 Eason <30045503+Eason0729@users.noreply.github.com> - 0.1.0
- Initial RPM package
- SELinux policy support for PAM integration
