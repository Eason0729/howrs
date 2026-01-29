#![feature(portable_simd)]

pub mod face;
pub mod model;
pub mod pipeline;
pub mod video;
pub mod yunet;

// Re-export commonly used types
pub use face::{Detection, Embedding};
pub use pipeline::Pipeline;
pub use video::Camera;
