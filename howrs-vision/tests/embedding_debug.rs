use anyhow::Result;
use howrs_vision::{face, model};
use ndarray::Array2;
use std::path::Path;

/// Test if the model is producing constant or near-constant embeddings
#[test]
fn test_embedding_distinctiveness() -> Result<()> {
    env_logger::try_init().ok();
    let mut detector = model::detector_session()?;
    let mut recognizer = model::recog_session()?;

    let test_images = [
        "test_faces/ir-cam/eason1.png",
        "test_faces/ir-cam/eason2.png",
        "test_faces/ir-cam/ling1.png",
        "test_faces/ir-cam/ling2.png",
    ];

    println!("\n=== Testing Embedding Distinctiveness ===\n");

    let mut embeddings = Vec::new();

    for img_path in &test_images {
        if !Path::new(img_path).exists() {
            continue;
        }

        let img = image::open(img_path)?;
        let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;

        if detections.is_empty() {
            continue;
        }

        let aligned = face::align_face(&img, &detections[0], 112)?;
        let embedding = face::encode_face(&mut recognizer, &aligned)?;

        let name = img_path.split('/').last().unwrap();
        println!("{:20} embedding:", name);

        // Print full embedding in compact format
        let vec: Vec<f32> = embedding.vector.iter().copied().collect();
        println!(
            "  Values: [{:.4}, {:.4}, {:.4}, ..., {:.4}, {:.4}, {:.4}]",
            vec[0],
            vec[1],
            vec[2],
            vec[vec.len() - 3],
            vec[vec.len() - 2],
            vec[vec.len() - 1]
        );

        // Compute L2 norm (should be ~1.0 after normalization)
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("  L2 Norm: {:.6}", norm);

        // Compute statistics
        let mean: f32 = vec.iter().sum::<f32>() / vec.len() as f32;
        let variance: f32 = vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vec.len() as f32;
        println!("  Mean: {:.6}, Variance: {:.6}", mean, variance);

        embeddings.push((name.to_string(), embedding));
    }

    // Compute pairwise similarities
    println!("\n=== Pairwise Cosine Similarities ===\n");
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let sim = face::match_embedding(&embeddings[i].1, &embeddings[j].1);
            let same_person =
                embeddings[i].0.starts_with("eason") == embeddings[j].0.starts_with("eason");
            let marker = if same_person { "(same)" } else { "(DIFF)" };
            println!(
                "{:15} <-> {:15}: {:.6} {}",
                embeddings[i].0, embeddings[j].0, sim, marker
            );
        }
    }

    // Compute element-wise correlation between embeddings
    println!("\n=== Element-wise Analysis ===\n");
    if embeddings.len() >= 2 {
        let emb1 = &embeddings[0].1.vector;
        let emb2 = &embeddings[2].1.vector; // Different person

        let vec1: Vec<f32> = emb1.iter().copied().collect();
        let vec2: Vec<f32> = emb2.iter().copied().collect();

        // Count how many dimensions have similar values
        let mut similar_count = 0;
        for i in 0..vec1.len() {
            if (vec1[i] - vec2[i]).abs() < 0.01 {
                similar_count += 1;
            }
        }

        println!("Comparing {} and {}:", embeddings[0].0, embeddings[2].0);
        println!(
            "  Dimensions with similar values (diff < 0.01): {}/{}",
            similar_count,
            vec1.len()
        );
        println!(
            "  Percentage: {:.1}%",
            similar_count as f32 / vec1.len() as f32 * 100.0
        );

        // Show a few dimensions with largest difference
        let mut diffs: Vec<(usize, f32)> = vec1
            .iter()
            .zip(vec2.iter())
            .enumerate()
            .map(|(i, (a, b))| (i, (a - b).abs()))
            .collect();
        diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\n  Top 10 dimensions with largest differences:");
        for (i, (dim, diff)) in diffs.iter().take(10).enumerate() {
            println!(
                "    Dim {}: diff = {:.4} (val1={:.4}, val2={:.4})",
                dim, diff, vec1[*dim], vec2[*dim]
            );
        }
    }

    Ok(())
}
