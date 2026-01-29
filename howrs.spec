Name:           howrs
Version:        0.1.0
Release:        1%{?dist}
Summary:        High-performance facial recognition authentication system for Linux
License:        Apache-2.0
URL:            https://github.com/Eason0729/howrs
Source0:        %{name}-%{version}.tar.gz

# Disable debuginfo package since we're using pre-built binaries
%global debug_package %{nil}

BuildRequires:  systemd-rpm-macros

Requires:       pam
Requires:       libv4l

%description
Howrs is a high-performance facial recognition authentication system written in
Rust, designed as a modern alternative to Howdy. It provides fast and secure
face-based authentication for Linux systems with PAM integration.

Features:
- Blazing fast with SIMD optimizations
- State-of-the-art YuNet face detection and SFace recognition
- Hardware acceleration support (OpenVINO, CUDA, etc.)
- Efficient binary storage format
- PAM integration for system authentication

%prep
%autosetup

%build
# Binaries are pre-built and included in the source tarball
# This allows packaging without requiring rust/cargo at build time
echo "Using pre-built binaries from target/release/"

%install
# Install binary
install -D -m 755 target/release/howrs %{buildroot}%{_sbindir}/howrs

# Install PAM module
install -D -m 755 target/release/libhowrs.so %{buildroot}/%{_lib}/security/libhowrs.so

# Install default configuration
install -D -m 644 packaging/config.toml %{buildroot}/usr/local/etc/howrs/config.toml

# Create face storage directory
install -d -m 700 %{buildroot}/usr/local/etc/howrs

# Install SELinux policy (if it exists and is not empty)
if [ -s packaging/howrs_pam.pp ]; then
    install -D -m 644 packaging/howrs_pam.pp %{buildroot}%{_datadir}/selinux/packages/%{name}/howrs_pam.pp
fi

%post
# Load SELinux policy module (if available)
if [ $1 -eq 1 ] ; then
    if [ -f %{_datadir}/selinux/packages/%{name}/howrs_pam.pp ]; then
        if command -v semodule > /dev/null 2>&1 && selinuxenabled 2>/dev/null; then
            echo "Installing SELinux policy module for howrs PAM integration..."
            semodule -i %{_datadir}/selinux/packages/%{name}/howrs_pam.pp 2>/dev/null || :
            # Restore file contexts
            restorecon -R %{_lib}/security/libhowrs.so 2>/dev/null || :
            restorecon -R %{_sbindir}/howrs 2>/dev/null || :
            restorecon -R /usr/local/etc/howrs 2>/dev/null || :
            echo "SELinux policy installed. Face data storage and camera access should work correctly."
        fi
    else
        echo "Note: SELinux policy module not included in this build."
        echo "      You may need to configure SELinux manually for camera access."
    fi
fi

cat <<EOF

================================================================================
HOWRS INSTALLATION COMPLETE
================================================================================

The howrs facial recognition system has been installed:

  Binary:      %{_sbindir}/howrs
  PAM module:  %{_lib}/security/libhowrs.so
  Config:      /usr/local/etc/howrs/config.toml

SELinux policy has been installed for PAM integration.

NEXT STEPS:

1. Enroll your face:
   $ sudo howrs enroll --user \$USER

2. Test authentication:
   $ sudo howrs test --user \$USER

3. Enable PAM authentication (MANUAL STEP):
   Edit /etc/pam.d/sudo or /etc/pam.d/system-auth and add:

   auth sufficient libhowrs.so

   Place this line BEFORE other auth methods for best experience.

4. Add your user to the video group for camera access:
   $ sudo usermod -a -G video \$USER

   Then logout and login again.

WARNING: Test PAM configuration carefully! Keep a root shell open when
         testing to avoid being locked out.

For more information, see the documentation at:
  https://github.com/Eason0729/howrs

================================================================================

EOF

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
* Thu Jan 30 2025 Howrs Team <team@howrs.example> - 0.1.0-1
- Initial RPM package
- SELinux policy support for PAM integration
- OpenVINO execution provider by default
