# Howrs Fedora RPM Packaging

This directory contains files for building Fedora RPM packages with SELinux support.

## Files

- **howrs.spec** - RPM spec file (in project root)
- **config.toml** - Default configuration file
- **howrs_pam.te** - SELinux policy type enforcement file
- **howrs_pam.fc** - SELinux file context definitions
- **build-selinux.sh** - Script to build SELinux policy module
- **build-rpm.sh** - Script to build the complete RPM package

## Prerequisites

Install required packages:

```bash
sudo dnf install rpm-build rpmdevtools selinux-policy-devel rust cargo clang
```

## Building the RPM

### Quick Build

```bash
./packaging/build-rpm.sh
```

This script will:
1. Build the SELinux policy module
2. Create a source tarball
3. Build the RPM package

### Manual Build

```bash
# Build SELinux policy
cd packaging
./build-selinux.sh
cd ..

# Setup RPM build environment
mkdir -p ~/rpmbuild/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Create source tarball
tar czf ~/rpmbuild/SOURCES/howrs-0.1.0.tar.gz \
    --exclude='target' --exclude='.git' \
    --transform 's,^,howrs-0.1.0/,' .

# Copy spec file
cp howrs.spec ~/rpmbuild/SPECS/

# Build package
rpmbuild -ba ~/rpmbuild/SPECS/howrs.spec
```

## Installation

```bash
# Install the package
sudo dnf install ~/rpmbuild/RPMS/x86_64/howrs-0.1.0-*.rpm

# The package will automatically:
# - Install binary to /usr/sbin/howrs
# - Install PAM module to /lib64/security/libhowrs.so
# - Install default config to /usr/local/etc/howrs/config.toml
# - Load SELinux policy module for PAM integration
# - Set correct SELinux contexts
```

## Package Contents

The RPM installs:

- `/usr/sbin/howrs` - Main binary for enrollment and testing
- `/lib64/security/libhowrs.so` - PAM module library
- `/usr/local/etc/howrs/config.toml` - Default configuration
- `/usr/share/selinux/packages/howrs/howrs_pam.pp` - SELinux policy module

## Post-Installation Steps

### 1. Add User to Video Group

```bash
sudo usermod -a -G video $USER
# Logout and login for changes to take effect
```

### 2. Enroll Face

```bash
sudo howrs enroll --user $USER
```

### 3. Test Authentication

```bash
sudo howrs test --user $USER
```

### 4. Enable PAM Authentication (Manual)

**Important:** The RPM does NOT automatically configure PAM. You must manually edit PAM configuration files.

For sudo authentication, edit `/etc/pam.d/sudo`:

```
#%PAM-1.0
auth       sufficient   libhowrs.so
auth       include      system-auth
account    include      system-auth
password   include      system-auth
session    include      system-auth
```

For system-wide authentication, edit `/etc/pam.d/system-auth`:

```
auth        sufficient    libhowrs.so
auth        required      pam_unix.so try_first_pass nullok
# ... rest of config
```

**Warning:** Always keep a root shell open when testing PAM changes to avoid being locked out!

## SELinux Policy

The package includes a SELinux policy module that allows:

- PAM module to access camera devices (`/dev/video*`)
- Reading/writing face data in `/usr/local/etc/howrs/`
- Reading configuration files
- Executing the howrs binary
- Logging to syslog

The policy is automatically loaded during installation and removed during uninstallation.

### SELinux Commands

```bash
# Check if policy is loaded
semodule -l | grep howrs

# Manually install policy
sudo semodule -i /usr/share/selinux/packages/howrs/howrs_pam.pp

# Remove policy
sudo semodule -r howrs_pam

# Check file contexts
ls -Z /lib64/security/libhowrs.so
ls -Z /usr/sbin/howrs
ls -Z /usr/local/etc/howrs/

# Restore file contexts
sudo restorecon -Rv /lib64/security/libhowrs.so
sudo restorecon -Rv /usr/sbin/howrs
sudo restorecon -Rv /usr/local/etc/howrs/
```

## Uninstallation

```bash
# Remove the package
sudo dnf remove howrs

# This will automatically:
# - Remove all installed files
# - Unload the SELinux policy module
# - NOT remove face data in /usr/local/etc/howrs/ (marked as config)
```

To completely remove including face data:

```bash
sudo dnf remove howrs
sudo rm -rf /usr/local/etc/howrs/
```

## Troubleshooting

### SELinux Denials

Check for SELinux denials:

```bash
sudo ausearch -m avc -ts recent | grep howrs
```

If you see denials, you may need to:

```bash
# Generate additional policy from audit log
sudo audit2allow -a -M howrs_custom

# Load the custom policy
sudo semodule -i howrs_custom.pp
```

### Build Failures

If the build fails:

```bash
# Clean RPM build directory
rm -rf ~/rpmbuild/BUILD/*

# Clean cargo build
cargo clean

# Retry build
./packaging/build-rpm.sh
```

## License

The packaging scripts and SELinux policy are provided under the same license as the main howrs project (MIT OR Apache-2.0).
