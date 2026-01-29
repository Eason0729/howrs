use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub static CONFIG_PATH: Lazy<&'static Path> = Lazy::new(|| {
    Path::new(option_env!("HOWRS_CONFIG_PATH").unwrap_or("/usr/local/etc/howrs/config.toml"))
});

pub static FACE_STORE_PREFIX: Lazy<&'static Path> = Lazy::new(|| {
    Path::new(option_env!("HOWRS_FACE_STORE_PREFIX").unwrap_or("/usr/local/etc/howrs"))
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub threshold: f32,
    pub camera: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            threshold: 0.6,
            camera: "/dev/video0".to_string(),
        }
    }
}

pub fn load_config(path: Option<&Path>) -> Result<Config> {
    let path = path.unwrap_or(&CONFIG_PATH);
    if !path.exists() {
        return Ok(Config::default());
    }
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading config at {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing config {}", path.display()))
}

pub fn save_config(cfg: &Config, path: Option<&Path>) -> Result<()> {
    let path = path.unwrap_or(&CONFIG_PATH);
    let data = toml::to_string_pretty(cfg)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, data)?;
    Ok(())
}
