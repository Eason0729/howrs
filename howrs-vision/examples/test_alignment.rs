use anyhow::Result;
use howrs_vision::{face, model};

fn main() -> Result<()> {
    env_logger::init();

    let mut detector = model::detector_session()?;
    let mut recognizer = model::recog_session()?;

    // Test with one of ling's images that the user mentioned
    let test_image = "test_faces/ir-cam/ling4.png";
    println!("Testing with {}", test_image);

    let img = image::open(test_image)?;
    println!("Original image: {}x{}", img.width(), img.height());

    let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;
    println!("Detected {} faces", detections.len());

    for (i, det) in detections.iter().enumerate() {
        println!("\nFace {}:", i);
        println!(
            "  Bbox: [{:.1}, {:.1}, {:.1}, {:.1}]",
            det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]
        );
        println!("  Score: {:.3}", det.score);
        println!("  Landmarks:");
        println!(
            "    Left eye: ({:.1}, {:.1})",
            det.landmarks[0], det.landmarks[1]
        );
        println!(
            "    Right eye: ({:.1}, {:.1})",
            det.landmarks[2], det.landmarks[3]
        );
        println!(
            "    Nose: ({:.1}, {:.1})",
            det.landmarks[4], det.landmarks[5]
        );
    }

    if let Some(det) = detections.first() {
        println!("\n=== Using Face 0 ===");
        println!("\nDetected face:");
        println!(
            "  Bbox: [{:.1}, {:.1}, {:.1}, {:.1}]",
            det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]
        );
        println!("  Landmarks:");
        println!(
            "    Left eye: ({:.1}, {:.1})",
            det.landmarks[0], det.landmarks[1]
        );
        println!(
            "    Right eye: ({:.1}, {:.1})",
            det.landmarks[2], det.landmarks[3]
        );
        println!(
            "    Nose: ({:.1}, {:.1})",
            det.landmarks[4], det.landmarks[5]
        );
        println!(
            "    Left mouth: ({:.1}, {:.1})",
            det.landmarks[6], det.landmarks[7]
        );
        println!(
            "    Right mouth: ({:.1}, {:.1})",
            det.landmarks[8], det.landmarks[9]
        );

        // Align and save
        let aligned = face::align_face(&img, det, 112)?;
        aligned.save("crop.png")?;
        println!("\n✓ Saved aligned face to crop.png");

        // Get embedding
        let embedding = face::encode_face(&mut recognizer, &aligned)?;
        println!(
            "✓ Generated embedding with {} dimensions",
            embedding.vector.len()
        );
    }

    Ok(())
}
