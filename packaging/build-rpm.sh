#!/bin/bash
# Build RPM package for howrs on Fedora
# This script packages howrs with SELinux support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VERSION="0.1.0"
PACKAGE_NAME="howrs-${VERSION}"

echo "==================================="
echo "Building howrs RPM package"
echo "==================================="

# Check if required tools are installed
echo "Checking prerequisites..."
if ! command -v rpmbuild &> /dev/null; then
    echo "Error: rpmbuild not found. Please install rpm-build:"
    echo "  sudo dnf install rpm-build rpmdevtools"
    exit 1
fi

if ! command -v cargo &> /dev/null; then
    echo "Error: cargo not found. Please install Rust:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Check if binaries already exist
echo "Checking for pre-built binaries..."
cd "$PROJECT_ROOT"
if [ -f target/release/howrs ] && [ -f target/release/libhowrs.so ]; then
    echo "Using existing binaries:"
    echo "  - target/release/howrs"
    echo "  - target/release/libhowrs.so"
else
    # Build the Rust project
    echo "Building Rust project..."
    export RUSTFLAGS="-C target-cpu=x86-64-v2 -C target-feature=+avx2"
    cargo build --release --features openvino
    if [ ! -f target/release/howrs ] || [ ! -f target/release/libhowrs.so ]; then
        echo "Error: Failed to build binaries"
        exit 1
    fi
    echo "Build complete: howrs and libhowrs.so"
fi

# Setup RPM build environment
echo "Setting up RPM build environment..."
mkdir -p ~/rpmbuild/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Build SELinux policy module (optional)
echo "Building SELinux policy module..."
cd "$SCRIPT_DIR"
if [ -f howrs_pam.pp ]; then
    echo "Using existing SELinux policy module: howrs_pam.pp"
elif command -v checkmodule &> /dev/null && rpm -q selinux-policy-devel &> /dev/null; then
    ./build-selinux.sh
else
    echo "WARNING: SELinux policy development tools not found."
    echo "  Install with: sudo dnf install selinux-policy-devel"
    echo "  Skipping SELinux policy build. The RPM will install without SELinux support."
    echo ""
    # Create a dummy policy file so the spec file doesn't fail
    touch howrs_pam.pp
fi

# Create source tarball with pre-built binaries
echo "Creating source tarball..."
cd "$PROJECT_ROOT"
TEMP_DIR="$HOME/rpmbuild/BUILD/howrs-temp-$$"
mkdir -p "${TEMP_DIR}/${PACKAGE_NAME}"

# Copy source files
echo "Copying project files..."
rsync -av \
    --exclude='.git' \
    --exclude='*.swp' \
    --exclude='*~' \
    --exclude='store' \
    --exclude='target' \
    . "${TEMP_DIR}/${PACKAGE_NAME}/"

# Copy only the built binaries
echo "Copying built binaries..."
mkdir -p "${TEMP_DIR}/${PACKAGE_NAME}/target/release"
cp target/release/howrs "${TEMP_DIR}/${PACKAGE_NAME}/target/release/"
cp target/release/libhowrs.so "${TEMP_DIR}/${PACKAGE_NAME}/target/release/"

# Create tarball
cd "${TEMP_DIR}"
tar czf "${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}"
mv "${PACKAGE_NAME}.tar.gz" ~/rpmbuild/SOURCES/
cd "$PROJECT_ROOT"

# Copy spec file
echo "Copying spec file..."
cp howrs.spec ~/rpmbuild/SPECS/

# Build RPM
echo "Building RPM package..."
rpmbuild -ba ~/rpmbuild/SPECS/howrs.spec

# Clean up
rm -rf "${TEMP_DIR}"

echo ""
echo "==================================="
echo "Build complete!"
echo "==================================="
echo ""
echo "RPM packages available at:"
echo "  Binary RPM: ~/rpmbuild/RPMS/x86_64/howrs-${VERSION}-*.rpm"
echo "  Source RPM: ~/rpmbuild/SRPMS/howrs-${VERSION}-*.src.rpm"
echo ""
echo "To install:"
echo "  sudo dnf install ~/rpmbuild/RPMS/x86_64/howrs-${VERSION}-*.rpm"
echo ""
