fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!(
        "cargo:rustc-link-arg-cdylib=-Wl,--dynamic-list={}/pam_dynamic_list.txt",
        dir
    );
    println!("cargo:rerun-if-changed=pam_dynamic_list.txt");

    println!("cargo:rustc-link-arg-cdylib=/lib64/libpam.so.0");
}
