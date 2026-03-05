fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        println!(
            "cargo:rustc-link-arg=-Wl,--dynamic-list={}/pam_dynamic_list.txt",
            dir
        );
        println!("cargo:rerun-if-changed=pam_dynamic_list.txt");

        println!("cargo:rustc-link-arg=/lib64/libpam.so.0");
    }
}
