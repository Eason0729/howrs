use crate::config::FACE_STORE_PREFIX;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::os::unix::fs::PermissionsExt;

#[derive(Debug, Serialize, Deserialize)]
pub struct FaceRecord {
    pub id: String,
    pub embedding: Vec<f32>,
}

fn user_store_path(user_id: &str) -> PathBuf {
    let mut p = FACE_STORE_PREFIX.to_path_buf();
    p.push(user_id);
    p
}

pub fn load_records(user_id: &str) -> Result<Vec<FaceRecord>> {
    let path = user_store_path(user_id);
    let file = path.join("faces.bin");
    
    if !file.exists() {
        return Ok(vec![]);
    }
    
    let data = std::fs::read(&file)
        .with_context(|| format!("reading {}", file.display()))?;
    Ok(postcard::from_bytes(&data)?)
}

pub fn save_record(user_id: &str, record: FaceRecord) -> Result<()> {
    let path = user_store_path(user_id);
    std::fs::create_dir_all(&path)?;
    // Set directory permissions to 755 (readable by all users, writable by root only)
    // This allows SDDM and other non-root display managers to read face data
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))?;
    
    let mut records = load_records(user_id)?;
    records.push(record);
    let file = path.join("faces.bin");
    let data = postcard::to_allocvec(&records)?;
    std::fs::write(&file, data)?;
    
    // Set file permissions to 644 (readable by all users, writable by root only)
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644))?;
    Ok(())
}

pub fn purge(user_id: &str) -> Result<()> {
    let path = user_store_path(user_id);
    if path.exists() {
        std::fs::remove_dir_all(&path).with_context(|| format!("removing {}", path.display()))?;
    }
    Ok(())
}
