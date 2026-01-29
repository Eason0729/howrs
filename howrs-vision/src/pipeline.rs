use anyhow::{Context, Result};
use image::DynamicImage;
use ort::session::Session;

use crate::face::{self, Detection, Embedding};

/// Full pipeline: detect faces → align → encode
pub struct Pipeline {
    pub detector: Session,
    pub encoder: Session,
}

impl Pipeline {
    pub fn new() -> Result<Self> {
        Ok(Self {
            detector: crate::model::detector_session()?,
            encoder: crate::model::recog_session()?,
        })
    }

    /// Process an image: detect best face and return embedding
    pub fn process_image(
        &mut self,
        img: &DynamicImage,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Detection, Embedding)> {
        // Detect faces
        let detections =
            face::detect_faces(&mut self.detector, img, score_threshold, nms_threshold)
                .context("detecting faces")?;

        if detections.is_empty() {
            anyhow::bail!("No face detected in image");
        }

        // Pick the best detection (highest score)
        let best = detections
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap();

        // Align and crop the face
        let face_img = face::align_face(img, best, 112).context("aligning face")?;

        // Encode to embedding
        let embedding = face::encode_face(&mut self.encoder, &face_img).context("encoding face")?;

        Ok((best.clone(), embedding))
    }

    /// Process and return only embedding (convenience method)
    pub fn extract_embedding(
        &mut self,
        img: &DynamicImage,
        _score_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Embedding> {
        let (_detection, embedding) = self.process_image(img, 0.6, nms_threshold)?;
        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let result = Pipeline::new();
        assert!(result.is_ok());
    }
}
