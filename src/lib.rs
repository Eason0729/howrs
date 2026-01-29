pub mod config;
pub mod identity;
pub mod matcher;
pub mod storage;

// Re-export vision types for convenience
pub use howrs_vision::{face, pipeline, video, Detection, Embedding, Pipeline};

// PAM module for cdylib
pub mod pam;
