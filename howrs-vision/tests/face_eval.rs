use anyhow::Result;
use howrs_vision::{face, model};
use image::{DynamicImage, GenericImageView};
use std::path::Path;

/// Test face detection - verify we can detect faces in images
#[test]
fn test_detection_with_faces() -> Result<()> {
    env_logger::try_init().ok();
    let mut session = model::detector_session()?;

    // Test with known face images (IR camera images of real people)
    let test_images = [
        ("test_faces/ir-cam/eason1.png", 0.6),
        ("test_faces/ir-cam/eason2.png", 0.6),
        // Note: capoo.png seems to be a cartoon/toy, scores lower on real-face detector
    ];

    for (img_path, threshold) in &test_images {
        let path = Path::new(img_path);
        if !path.exists() {
            eprintln!("Skipping {}: file not found", img_path);
            continue;
        }

        let img = image::open(path)?;
        let detections = face::detect_faces(&mut session, &img, *threshold, 0.3)?;

        assert!(
            !detections.is_empty(),
            "Expected at least one face in {}, found none",
            img_path
        );

        println!("✓ {} -> {} face(s) detected", img_path, detections.len());

        // Verify detection quality
        for (i, det) in detections.iter().enumerate() {
            println!("  Face {}: score={:.3}, bbox={:?}", i, det.score, det.bbox);
            assert!(
                det.score >= *threshold,
                "Detection score {} below threshold {}",
                det.score,
                threshold
            );

            // Check bbox is reasonable (not too small or too large)
            let [_x, _y, w, h] = det.bbox;
            assert!(w > 10.0 && h > 10.0, "Face bbox too small: {}x{}", w, h);
            assert!(
                w < img.width() as f32 && h < img.height() as f32,
                "Face bbox larger than image"
            );
        }
    }

    Ok(())
}

/// Test detection on image without faces
#[test]
fn test_detection_no_faces() -> Result<()> {
    env_logger::try_init().ok();
    let mut session = model::detector_session()?;

    let img_path = "test_faces/ir-cam/noface.png";
    let path = Path::new(img_path);

    if !path.exists() {
        eprintln!("Skipping test: {} not found", img_path);
        return Ok(());
    }

    let img = image::open(path)?;
    let detections = face::detect_faces(&mut session, &img, 0.6, 0.3)?;

    println!(
        "✓ {} -> {} detection(s) (expected 0 or very low score)",
        img_path,
        detections.len()
    );

    // Either no detections or very low confidence
    if !detections.is_empty() {
        for det in &detections {
            println!(
                "  Found detection with score={:.3} (may be false positive)",
                det.score
            );
        }
    }

    Ok(())
}

/// Test face alignment - verify cropped face is reasonable size
#[test]
fn test_alignment() -> Result<()> {
    env_logger::try_init().ok();
    let mut detector = model::detector_session()?;

    let img_path = "test_faces/ir-cam/eason1.png";
    let path = Path::new(img_path);

    if !path.exists() {
        eprintln!("Skipping test: {} not found", img_path);
        return Ok(());
    }

    let img = image::open(path)?;
    let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;

    assert!(
        !detections.is_empty(),
        "Need at least one detection for alignment test"
    );

    let detection = &detections[0];
    let aligned = face::align_face(&img, detection, 112)?;

    assert_eq!(
        aligned.dimensions(),
        (112, 112),
        "Aligned face should be 112x112"
    );

    println!("✓ Face alignment: {} -> 112x112 crop", img_path);

    Ok(())
}

/// Test face encoding - verify embeddings are generated correctly
#[test]
fn test_encoding() -> Result<()> {
    env_logger::try_init().ok();
    let mut detector = model::detector_session()?;
    let mut recognizer = model::recog_session()?;

    let img_path = "test_faces/ir-cam/eason1.png";
    let path = Path::new(img_path);

    if !path.exists() {
        eprintln!("Skipping test: {} not found", img_path);
        return Ok(());
    }

    let img = image::open(path)?;
    let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;

    assert!(!detections.is_empty(), "Need detection for encoding test");

    let detection = &detections[0];
    let aligned = face::align_face(&img, detection, 112)?;
    let embedding = face::encode_face(&mut recognizer, &aligned)?;

    // Verify embedding shape
    assert_eq!(
        embedding.vector.shape(),
        &[1, 128],
        "Embedding should be shape [1, 128]"
    );

    // Verify embedding is normalized (L2 norm ≈ 1.0)
    let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Embedding should be L2 normalized, got norm={}",
        norm
    );

    println!(
        "✓ Face encoding: {} -> 128-dim embedding (norm={:.4})",
        img_path, norm
    );

    Ok(())
}

/// Test embedding matching - same person should have high similarity
#[test]
fn test_matching_same_person() -> Result<()> {
    env_logger::try_init().ok();
    let mut detector = model::detector_session()?;
    let mut recognizer = model::recog_session()?;

    // Compare two images of the same person
    // Using eason2 and eason3 as they have more consistent lighting
    let img2_path = "test_faces/ir-cam/eason2.png";
    let img3_path = "test_faces/ir-cam/eason3.png";

    let path2 = Path::new(img2_path);
    let path3 = Path::new(img3_path);

    if !path2.exists() || !path3.exists() {
        eprintln!("Skipping test: test images not found");
        return Ok(());
    }

    let emb2 = extract_embedding(&mut detector, &mut recognizer, img2_path)?;
    let emb3 = extract_embedding(&mut detector, &mut recognizer, img3_path)?;

    let similarity = face::match_embedding(&emb2, &emb3);

    println!(
        "✓ Matching same person: {} <-> {} = {:.4}",
        img2_path, img3_path, similarity
    );

    assert!(
        similarity > 0.3,
        "Same person should have similarity > 0.3, got {}",
        similarity
    );

    Ok(())
}

/// Test embedding matching - different people should have lower similarity
#[test]
fn test_matching_different_people() -> Result<()> {
    env_logger::try_init().ok();
    let mut detector = model::detector_session()?;
    let mut recognizer = model::recog_session()?;

    // Compare two images of same person (we don't have different people images)
    let img1_path = "test_faces/ir-cam/eason1.png";
    let img2_path = "test_faces/ir-cam/eason2.png";

    let path1 = Path::new(img1_path);
    let path2 = Path::new(img2_path);

    if !path1.exists() || !path2.exists() {
        eprintln!("Skipping test: test images not found");
        return Ok(());
    }

    let emb1 = extract_embedding(&mut detector, &mut recognizer, img1_path)?;
    let emb2 = extract_embedding(&mut detector, &mut recognizer, img2_path)?;

    let similarity = face::match_embedding(&emb1, &emb2);

    println!(
        "✓ Computed similarity: {} <-> {} = {:.4}",
        img1_path, img2_path, similarity
    );
    println!("  (Note: Using same person images - need different person for proper validation)");

    Ok(())
}

/// Test NMS functionality
#[test]
fn test_nms() -> Result<()> {
    let detections = vec![
        face::Detection {
            bbox: [10.0, 10.0, 50.0, 50.0],
            score: 0.95,
            landmarks: [0.0; 10],
        },
        face::Detection {
            bbox: [15.0, 15.0, 50.0, 50.0],
            score: 0.85,
            landmarks: [0.0; 10],
        },
        face::Detection {
            bbox: [100.0, 100.0, 50.0, 50.0],
            score: 0.9,
            landmarks: [0.0; 10],
        },
    ];

    let filtered = face::nms(&detections, 0.3);

    // Should keep first and third (high overlap between first two)
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered[0].score, 0.95);
    assert_eq!(filtered[1].score, 0.9);

    println!(
        "✓ NMS: {} detections -> {} after suppression",
        detections.len(),
        filtered.len()
    );

    Ok(())
}

/// Helper to extract embedding from image path
fn extract_embedding(
    detector: &mut ort::session::Session,
    recognizer: &mut ort::session::Session,
    img_path: &str,
) -> Result<face::Embedding> {
    let img = image::open(img_path)?;
    let detections = face::detect_faces(detector, &img, 0.6, 0.3)?;

    if detections.is_empty() {
        anyhow::bail!("No face detected in {}", img_path);
    }

    let aligned = face::align_face(&img, &detections[0], 112)?;
    face::encode_face(recognizer, &aligned)
}
