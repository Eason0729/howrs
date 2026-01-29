use anyhow::Result;
use howrs_vision::{face, model};
use image::GenericImageView;
use std::path::Path;

/// Test if channel order matters (RGB vs BGR)
#[test]
fn test_channel_order() -> Result<()> {
    env_logger::try_init().ok();
    let mut detector = model::detector_session()?;
    let mut recognizer = model::recog_session()?;

    let img_path = "test_faces/ir-cam/eason1.png";
    if !Path::new(img_path).exists() {
        eprintln!("Skipping: {} not found", img_path);
        return Ok(());
    }

    let img = image::open(img_path)?;
    let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;

    if detections.is_empty() {
        eprintln!("No faces detected");
        return Ok(());
    }

    let aligned = face::align_face(&img, &detections[0], 112)?;

    // Get embedding with current implementation
    let embedding_rgb = face::encode_face(&mut recognizer, &aligned)?;

    println!("Embedding (RGB order):");
    print!("  First 10: [");
    for i in 0..10 {
        print!("{:.4}", embedding_rgb.vector[[0, i]]);
        if i < 9 {
            print!(", ");
        }
    }
    println!("]");

    // Check if all channels are the same (grayscale detection)
    let rgb = aligned.to_rgb8();
    let mut all_same = true;
    for pixel in rgb.pixels() {
        if pixel[0] != pixel[1] || pixel[1] != pixel[2] {
            all_same = false;
            break;
        }
    }

    if all_same {
        println!("\n⚠ WARNING: Input image has identical RGB channels (grayscale)");
        println!("  This might cause the SFace model to produce similar embeddings for all faces!");
    }

    Ok(())
}
