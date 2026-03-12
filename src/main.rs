use libloading::Library;

fn main() {
    let path = std::env::var("HOWRS_LIBRARY_PATH").unwrap_or_else(|_| "libhowrs.so".to_string());

    unsafe {
        let lib = match Library::new(&path) {
            Ok(lib) => lib,
            Err(e) => {
                eprintln!("howrs: failed to load library: {}\nHint: set HOWRS_LIBRARY_PATH to the .so path", e);
                std::process::exit(1);
            }
        };

        let func: libloading::Symbol<unsafe extern "C" fn() -> libc::c_int> =
            match lib.get(b"howrs_main\0") {
                Ok(func) => func,
                Err(e) => {
                    eprintln!("howrs: failed to find howrs_main symbol: {}", e);
                    std::process::exit(1);
                }
            };

        std::process::exit(func() as i32);
    }
}
