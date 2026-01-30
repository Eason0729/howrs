#!/bin/bash
# Build source RPM using fedpkg with SELinux support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Project information
PROJECT_NAME="howrs"
VERSION="0.1.0"
RELEASE="1"

echo "=== Building source RPM for $PROJECT_NAME using fedpkg ==="
echo "Project root: $PROJECT_ROOT"
echo "Version: $VERSION"
echo "Release: $RELEASE"
echo ""

# Check if fedpkg is installed
if ! command -v fedpkg &> /dev/null; then
    echo "Error: fedpkg not found. Please install it:"
    echo "  sudo dnf install fedpkg"
    exit 1
fi

# Build SELinux policy module first
echo "Step 1: Downloading ONNX models..."
cd "$PROJECT_ROOT/howrs-vision/models"
if [ -f download_models.sh ]; then
    bash download_models.sh
else
    echo "Error: download_models.sh not found"
    exit 1
fi

# Verify models were downloaded
if [ ! -f face_detection_yunet_2023mar.onnx ] || [ ! -f face_recognition_sface_2021dec.onnx ]; then
    echo "Error: ONNX models not found after download"
    exit 1
fi

echo "ONNX models downloaded successfully"

echo ""
echo "Step 2: Building SELinux policy module..."
cd "$SCRIPT_DIR"
if [ -f build-selinux.sh ]; then
    bash build-selinux.sh
else
    echo "Warning: build-selinux.sh not found, skipping SELinux policy build"
fi

# Verify SELinux policy was built
if [ ! -f "$SCRIPT_DIR/howrs_pam.pp" ]; then
    echo "Warning: SELinux policy module (howrs_pam.pp) not found"
    echo "The SRPM will be built without SELinux support"
fi

cd "$PROJECT_ROOT"

# Create a clean source tarball
echo ""
echo "Step 3: Creating source tarball..."
TARBALL_NAME="${PROJECT_NAME}-${VERSION}.tar.gz"
TEMP_DIR=$(mktemp -d)
SOURCE_DIR="$TEMP_DIR/${PROJECT_NAME}-${VERSION}"

# Copy project files to temporary directory
mkdir -p "$SOURCE_DIR"
rsync -a --exclude='.git' \
         --exclude='target' \
         --exclude='*.swp' \
         --exclude='*~' \
         --exclude='.gitignore' \
         --exclude='rpmbuild' \
         "$PROJECT_ROOT/" "$SOURCE_DIR/"

# Create tarball
cd "$TEMP_DIR"
tar czf "$TARBALL_NAME" "${PROJECT_NAME}-${VERSION}"
mv "$TARBALL_NAME" "$PROJECT_ROOT/"
cd "$PROJECT_ROOT"

# Clean up temp directory
rm -rf "$TEMP_DIR"

echo "Created source tarball: $TARBALL_NAME"

# Prepare fedpkg environment
echo ""
echo "Step 4: Setting up fedpkg environment..."
FEDPKG_DIR="$PROJECT_ROOT/fedpkg"
rm -rf "$FEDPKG_DIR"
mkdir -p "$FEDPKG_DIR"
cd "$FEDPKG_DIR"

# Copy spec file
cp "$SCRIPT_DIR/howrs.spec" .

# Update spec file to use the correct Source0
sed -i "s|Source0:.*|Source0: ${TARBALL_NAME}|" howrs.spec

# Copy source tarball to fedpkg directory
cp "$PROJECT_ROOT/$TARBALL_NAME" .

# Create sources file for fedpkg
echo "SHA512 (${TARBALL_NAME}) = $(sha512sum ${TARBALL_NAME} | awk '{print $1}')" > sources

# Build source RPM using fedpkg
echo ""
echo "Step 5: Building source RPM with fedpkg..."
fedpkg --release f40 srpm

# Find and report the built SRPM
SRPM_FILE=$(find . -name "*.src.rpm" | head -n 1)
if [ -n "$SRPM_FILE" ]; then
    # Copy SRPM to project root for easy access
    cp "$SRPM_FILE" "$PROJECT_ROOT/"
    SRPM_NAME=$(basename "$SRPM_FILE")
    
    echo ""
    echo "=== Success! ==="
    echo "Source RPM created: $PROJECT_ROOT/$SRPM_NAME"
    echo ""
    echo "To build the binary RPM from this SRPM:"
    echo "  rpmbuild --rebuild $SRPM_NAME"
    echo ""
    echo "Or use mock for a clean build:"
    echo "  mock -r fedora-40-x86_64 $SRPM_NAME"
    echo ""
    echo "The SRPM includes:"
    echo "  - Source code"
    echo "  - Spec file"
    echo "  - ONNX models (for offline build)"
    if [ -f "$SCRIPT_DIR/howrs_pam.pp" ]; then
        echo "  - SELinux policy module (howrs_pam.pp)"
    fi
else
    echo "Error: Failed to find built SRPM"
    exit 1
fi

# Optional: Clean up fedpkg directory
# Uncomment the following line if you want to remove the temporary fedpkg directory
# rm -rf "$FEDPKG_DIR"

echo ""
echo "Fedpkg directory kept at: $FEDPKG_DIR"
echo "You can remove it manually if not needed: rm -rf $FEDPKG_DIR"
