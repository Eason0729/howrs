use anyhow::{Context, Result};
use ort::{
    ep,
    session::{builder::GraphOptimizationLevel, Session},
};

// Placeholder: include_bytes for required models. In a real setup, these would be the actual files.
pub static FACE_RECOGNITION_MODEL: &[u8] =
    include_bytes!("../models/face_recognition_sface_2021dec.onnx");
pub static DETECTOR_MODEL: &[u8] = include_bytes!("../models/face_detection_yunet_2023mar.onnx");

pub fn recog_session() -> Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_execution_providers([
            #[cfg(feature = "openvino")]
            ep::OpenVINO::default().into(),
            #[cfg(feature = "cuda")]
            ep::CUDA::default().into(),
            ep::CPU::default().into(),
        ])?
        .commit_from_memory(FACE_RECOGNITION_MODEL)
        .context("load recognition model")
}

pub fn detector_session() -> Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_execution_providers([
            #[cfg(feature = "openvino")]
            ep::OpenVINO::default().into(),
            #[cfg(feature = "cuda")]
            ep::CUDA::default().into(),
            ep::CPU::default().into(),
        ])?
        .commit_from_memory(DETECTOR_MODEL)
        .context("load detector model")
}
