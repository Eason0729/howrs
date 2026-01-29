//! Check: where are landmarks positioned relative to image dimensions?
use howrs_vision::{face, model};

#[test]
#[ignore]
fn analyze_landmark_positions() {
    let mut detector = model::detector_session().expect("Failed to load model");

    let img_path = "test_faces/ir-cam/eason1.png";
    let img = image::open(img_path).expect("Failed to load image");
    let (img_w, img_h) = (img.width(), img.height());
    println!("Image: {} ({}x{})", img_path, img_w, img_h);

    let detections =
        face::detect_faces(&mut detector, &img, 0.6, 0.3).expect("Failed to detect faces");

    println!("\nFound {} detections\n", detections.len());

    for (idx, det) in detections.iter().enumerate() {
        println!("Detection {}:", idx);
        println!(
            "  BBox: [{:.1}, {:.1}, {:.1}, {:.1}] (x, y, w, h)",
            det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]
        );

        let landmark_names = ["LeftEye", "RightEye", "Nose", "LeftMouth", "RightMouth"];

        println!("  Landmarks:");
        for i in 0..5 {
            let x = det.landmarks[i * 2];
            let y = det.landmarks[i * 2 + 1];
            let x_pct = (x / img_w as f32 * 100.0) as i32;
            let y_pct = (y / img_h as f32 * 100.0) as i32;
            println!(
                "    {}: ({:.1}, {:.1}) = ({}%, {}%)",
                landmark_names[i], x, y, x_pct, y_pct
            );
        }

        // Calculate eye positions
        let left_eye_y = det.landmarks[1];
        let right_eye_y = det.landmarks[3];
        let eye_center_y = (left_eye_y + right_eye_y) / 2.0;
        let eye_y_pct = (eye_center_y / img_h as f32 * 100.0) as i32;

        println!("\n  Analysis:");
        println!(
            "    Eye center Y: {:.1} ({}% from top)",
            eye_center_y, eye_y_pct
        );
        println!("    Expected for typical face: 30-40% from top");

        if eye_y_pct > 50 {
            println!("    ⚠️  WARNING: Eyes are in lower half! Face is likely cut off at top!");
        } else {
            println!("    ✓ Eyes are in upper half - landmarks seem reasonable");
        }
    }
}
