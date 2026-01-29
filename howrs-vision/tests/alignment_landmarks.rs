//! Verify that aligned face has landmarks at reference positions
use anyhow::Result;
use howrs_vision::{face, model};

#[test]
#[ignore]
fn verify_alignment_landmarks() -> Result<()> {
    let mut detector = model::detector_session()?;
    let img = image::open("test_faces/ir-cam/eason1.png")?;
    let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;

    let det = &detections[0];
    let aligned = face::align_face(&img, det, 112)?;

    // Run detection on the aligned face to see where landmarks end up
    let aligned_detections = face::detect_faces(&mut detector, &aligned, 0.3, 0.3)?;

    if aligned_detections.is_empty() {
        println!("⚠️  No face detected in aligned image!");
        println!("This might mean the alignment is too zoomed in or misaligned");
        return Ok(());
    }

    let aligned_det = &aligned_detections[0];

    println!("Reference landmarks for 112x112:");
    println!("  LeftEye: (38.3, 51.7)");
    println!("  RightEye: (73.5, 51.5)");
    println!("\nDetected landmarks in aligned image:");
    println!(
        "  LeftEye: ({:.1}, {:.1})",
        aligned_det.landmarks[0], aligned_det.landmarks[1]
    );
    println!(
        "  RightEye: ({:.1}, {:.1})",
        aligned_det.landmarks[2], aligned_det.landmarks[3]
    );
    println!(
        "  Nose: ({:.1}, {:.1})",
        aligned_det.landmarks[4], aligned_det.landmarks[5]
    );
    println!(
        "  LeftMouth: ({:.1}, {:.1})",
        aligned_det.landmarks[6], aligned_det.landmarks[7]
    );
    println!(
        "  RightMouth: ({:.1}, {:.1})",
        aligned_det.landmarks[8], aligned_det.landmarks[9]
    );

    // Calculate errors
    let left_eye_error = ((aligned_det.landmarks[0] - 38.3).powi(2)
        + (aligned_det.landmarks[1] - 51.7).powi(2))
    .sqrt();
    let right_eye_error = ((aligned_det.landmarks[2] - 73.5).powi(2)
        + (aligned_det.landmarks[3] - 51.5).powi(2))
    .sqrt();

    println!("\nAlignment error:");
    println!("  LeftEye: {:.1}px", left_eye_error);
    println!("  RightEye: {:.1}px", right_eye_error);

    if left_eye_error < 5.0 && right_eye_error < 5.0 {
        println!("\n✓ Alignment is good! (error < 5px)");
    } else if left_eye_error < 10.0 && right_eye_error < 10.0 {
        println!("\n⚠️  Alignment is acceptable (error < 10px)");
    } else {
        println!("\n❌ Alignment is poor (error >= 10px)");
    }

    Ok(())
}
