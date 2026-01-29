use anyhow::Result;
use howrs_vision::{face, model};
use image::GenericImageView;
use std::path::Path;

/// Debug test to inspect face alignment and cropping
#[test]
#[ignore = "debug_script"]
fn debug_face_alignment() -> Result<()> {
    env_logger::try_init().ok();
    let mut detector = model::detector_session()?;

    let test_images = [
        "test_faces/ir-cam/eason1.png",
        "test_faces/ir-cam/ling1.png",
    ];

    println!("\n=== Debugging Face Alignment ===");

    for img_path in &test_images {
        if !Path::new(img_path).exists() {
            continue;
        }

        let img = image::open(img_path)?;
        let (orig_w, orig_h) = img.dimensions();

        println!("\n{}", img_path);
        println!("  Original size: {}x{}", orig_w, orig_h);

        let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;

        if detections.is_empty() {
            println!("  No faces detected!");
            continue;
        }

        for (i, det) in detections.iter().enumerate() {
            println!("\n  Detection {}:", i);
            println!("    Score: {:.4}", det.score);
            println!(
                "    BBox: [{:.1}, {:.1}, {:.1}, {:.1}] (x, y, w, h)",
                det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]
            );

            // Print landmarks
            println!("    Landmarks:");
            for j in 0..5 {
                println!(
                    "      Point {}: ({:.1}, {:.1})",
                    j + 1,
                    det.landmarks[j * 2],
                    det.landmarks[j * 2 + 1]
                );
            }

            // Compute alignment crop
            let [x, y, w, h] = det.bbox;
            let padding = 0.3;
            let expanded_w = w * (1.0 + 2.0 * padding);
            let expanded_h = h * (1.0 + 2.0 * padding);
            let crop_size = expanded_w.max(expanded_h);
            let center_x = x + w / 2.0;
            let center_y = y + h / 2.0;

            let x1 = (center_x - crop_size / 2.0).max(0.0) as u32;
            let y1 = (center_y - crop_size / 2.0).max(0.0) as u32;
            let x2 = ((center_x + crop_size / 2.0).min(orig_w as f32)) as u32;
            let y2 = ((center_y + crop_size / 2.0).min(orig_h as f32)) as u32;

            println!(
                "    Alignment crop: [{}, {}, {}, {}] (x1, y1, x2, y2)",
                x1, y1, x2, y2
            );
            println!("    Crop size: {}x{}", x2 - x1, y2 - y1);

            // Check if crop is at image boundary
            let at_left = x1 == 0;
            let at_top = y1 == 0;
            let at_right = x2 == orig_w;
            let at_bottom = y2 == orig_h;

            if at_left || at_top || at_right || at_bottom {
                println!("    ⚠ WARNING: Crop hits image boundary!");
                if at_left {
                    println!("      - Crop at left edge");
                }
                if at_top {
                    println!("      - Crop at top edge");
                }
                if at_right {
                    println!("      - Crop at right edge");
                }
                if at_bottom {
                    println!("      - Crop at bottom edge");
                }
            }

            // Save aligned face for inspection
            let aligned = face::align_face(&img, det, 112)?;
            let output_name = format!(
                "debug_aligned_{}_{}.png",
                img_path.split('/').last().unwrap().replace(".png", ""),
                i
            );
            aligned.save(&output_name)?;
            println!("    Saved aligned face to: {}", output_name);
        }
    }

    Ok(())
}

/// Debug test to check if all embeddings are similar
#[test]
#[ignore = "debug_script"]
fn debug_embedding_variance() -> Result<()> {
    env_logger::try_init().ok();
    let mut detector = model::detector_session()?;
    let mut recognizer = model::recog_session()?;

    let test_images = [
        "test_faces/ir-cam/eason1.png",
        "test_faces/ir-cam/ling1.png",
    ];

    println!("\n=== Debugging Embedding Variance ===");

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

        println!("\n{}", img_path);
        println!("  Embedding shape: {:?}", embedding.vector.shape());

        // Print first 10 values
        print!("  First 10 values: [");
        for i in 0..10.min(embedding.vector.len()) {
            print!("{:.4}", embedding.vector[[0, i]]);
            if i < 9 {
                print!(", ");
            }
        }
        println!("]");

        // Compute statistics
        let values: Vec<f32> = embedding.vector.iter().copied().collect();
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        let variance: f32 =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        println!("  Mean: {:.4}", mean);
        println!("  Std Dev: {:.4}", std_dev);
        println!(
            "  Min: {:.4}",
            values.iter().cloned().fold(f32::INFINITY, f32::min)
        );
        println!(
            "  Max: {:.4}",
            values.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // Check for zero or near-zero variance (bad embedding)
        if std_dev < 0.01 {
            println!("  ⚠ WARNING: Very low variance - embeddings might be degenerate!");
        }
    }

    Ok(())
}
