use crate::{storage::FaceRecord, Embedding};

pub fn best_score(records: &[FaceRecord], probe: &Embedding) -> Option<f32> {
    records
        .iter()
        .map(|r| {
            let emb = Embedding {
                vector: ndarray::Array2::from_shape_vec(
                    (1, r.embedding.len()),
                    r.embedding.clone(),
                )
                .unwrap_or_else(|_| ndarray::Array2::zeros((1, 128))),
            };
            match_embedding(&emb, probe)
        })
        .fold(None, |acc, s| match acc {
            Some(best) if best > s => Some(best),
            _ => Some(s),
        })
}

pub fn match_embedding(a: &Embedding, b: &Embedding) -> f32 {
    howrs_vision::face::match_embedding(a, b)
}
