use anyhow::Result;
use libc::{getpwuid, uid_t};
use std::ffi::CStr;

pub fn current_user_id() -> Result<String> {
    if let Ok(sudo_uid) = std::env::var("SUDO_UID") {
        return Ok(sudo_uid);
    }
    unsafe {
        let uid = libc::geteuid();
        let pwd = getpwuid(uid as uid_t);
        if pwd.is_null() {
            return Err(anyhow::anyhow!("failed to resolve current user"));
        }
        let name = CStr::from_ptr((*pwd).pw_name);
        Ok(name.to_string_lossy().into_owned())
    }
}
