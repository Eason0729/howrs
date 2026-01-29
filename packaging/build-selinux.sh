#!/bin/bash
# Build SELinux policy module for howrs PAM integration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building SELinux policy module for howrs..."

# Check if SELinux policy development tools are available
if ! command -v make &> /dev/null; then
    echo "Error: make not found. Please install make package:"
    echo "  sudo dnf install make"
    exit 1
fi

if [ ! -f /usr/share/selinux/devel/Makefile ]; then
    echo "Error: SELinux development Makefile not found. Please install selinux-policy-devel package:"
    echo "  sudo dnf install selinux-policy-devel"
    exit 1
fi

# Clean any previous builds
rm -f howrs_pam.pp howrs_pam.mod
rm -rf tmp/

# Build the policy module using the SELinux devel Makefile
echo "Compiling policy module..."
make -f /usr/share/selinux/devel/Makefile howrs_pam.pp

if [ -f howrs_pam.pp ]; then
    echo "SELinux policy module built successfully: howrs_pam.pp"
    echo ""
    echo "To install manually:"
    echo "  sudo semodule -i howrs_pam.pp"
    echo ""
    echo "To check if installed:"
    echo "  semodule -l | grep howrs"
    echo ""
    echo "To remove:"
    echo "  sudo semodule -r howrs_pam"
else
    echo "Error: Failed to build howrs_pam.pp"
    exit 1
fi
