use anyhow::Result;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::time::{Duration, Instant};

// PAM return codes
const PAM_SUCCESS: c_int = 0;
const PAM_AUTH_ERR: c_int = 7;
const PAM_USER_UNKNOWN: c_int = 10;
const PAM_SYSTEM_ERR: c_int = 4;

// PAM item types
const PAM_USER: c_int = 2;

// PAM handle opaque pointer type
type PamHandle = c_void;

// External PAM function we need
extern "C" {
    fn pam_get_item(pamh: *const PamHandle, item_type: c_int, item: *mut *const c_void) -> c_int;
}

#[no_mangle]
pub extern "C" fn pam_sm_authenticate(
    pamh: *mut PamHandle,
    _flags: c_int,
    _argc: c_int,
    _argv: *const *const c_char,
) -> c_int {
    // Get username from PAM
    let username = match get_pam_user(pamh) {
        Ok(user) => user,
        Err(_) => return PAM_USER_UNKNOWN,
    };

    eprintln!("Running facial recognition...");

    // Run authentication
    match run_auth(&username) {
        Ok(true) => PAM_SUCCESS,
        Ok(false) => PAM_AUTH_ERR,
        Err(_) => PAM_SYSTEM_ERR,
    }
}

#[no_mangle]
pub extern "C" fn pam_sm_setcred(
    _pamh: *mut PamHandle,
    _flags: c_int,
    _argc: c_int,
    _argv: *const *const c_char,
) -> c_int {
    return PAM_SUCCESS;
}

fn get_pam_user(pamh: *mut PamHandle) -> Result<String> {
    unsafe {
        let mut user_ptr: *const c_void = std::ptr::null();
        let ret = pam_get_item(pamh, PAM_USER, &mut user_ptr as *mut *const c_void);
        if ret != PAM_SUCCESS || user_ptr.is_null() {
            return Err(anyhow::anyhow!("Failed to get PAM user"));
        }
        let user_cstr = CStr::from_ptr(user_ptr as *const c_char);
        Ok(user_cstr.to_string_lossy().into_owned())
    }
}

fn run_auth(username: &str) -> Result<bool> {
    let config = crate::config::load_config(None)?;

    let records = crate::storage::load_records(username)?;
    if records.is_empty() {
        return Ok(false);
    }

    let mut pipeline = crate::Pipeline::new()?;

    use howrs_vision::Camera;
    let mut camera = Camera::open(&config.camera)?;

    let start_time = Instant::now();
    let scan_duration = Duration::from_secs(config.scan_durnation as u64);

    while start_time.elapsed() < scan_duration {
        if let Ok(frame_buf) = camera.frame() {
            let img = image::DynamicImage::ImageRgb8(frame_buf);
            // Use lower thresholds for faster processing in PAM context
            if let Ok(embedding) = pipeline.extract_embedding(&img, 0.5, 0.3) {
                let score = crate::matcher::best_score(&records, &embedding)
                    .ok_or_else(|| anyhow::anyhow!("No match found"))?;

                if score >= config.threshold {
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}
