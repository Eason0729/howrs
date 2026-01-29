use anyhow::{Context, Result};
use ort::{
    ep::{self, ExecutionProvider},
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session,
    },
};

// Placeholder: include_bytes for required models. In a real setup, these would be the actual files.
pub static FACE_RECOGNITION_MODEL: &[u8] =
    include_bytes!("../models/face_recognition_sface_2021dec.onnx");
pub static DETECTOR_MODEL: &[u8] = include_bytes!("../models/face_detection_yunet_2023mar.onnx");

pub fn session_builder() -> Result<SessionBuilder> {
    let mut builder =
        Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;

    #[cfg(feature = "openvino")]
    {
        let ep = ep::OpenVINO::default();
        if ep.is_available()? {
            ep.register(&mut builder)?;
        } else {
            log::warn!("openvino feature is enabled, onnx runtime not compiled with openvino")
        }
    }

    #[cfg(feature = "cuda")]
    {
        let ep = ep::CUDA::default();
        if ep.is_available()? {
            ep.register(&mut builder);
        } else {
            log::warn!("cuda feature is enabled, onnx runtime not compiled with cuda")
        }
    }

    Ok(builder)
}

pub fn recog_session() -> Result<Session> {
    session_builder()?
        .commit_from_memory(FACE_RECOGNITION_MODEL)
        .context("load recognition model")
}

pub fn detector_session() -> Result<Session> {
    session_builder()?
        .commit_from_memory(DETECTOR_MODEL)
        .context("load detector model")
}
