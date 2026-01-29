use anyhow::Result;
use howrs_vision::{face, model};
use image::GenericImageView;

/// Test to visualize where landmark positions map to in the crop
#[test]
fn test_landmark_mapping() -> Result<()> {
    let mut detector = model::detector_session()?;

    let img_path = "test_faces/ir-cam/eason1.png";
    println!("\n=== Testing landmark mapping for {} ===", img_path);

    let img = image::open(img_path)?;
    let (width, height) = img.dimensions();
    println!("Original image: {}x{}", width, height);

    let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;
    let det = &detections[0];

    println!("\nDetected landmarks in original image:");
    let landmark_names = ["left eye", "right eye", "nose", "left mouth", "right mouth"];
    for i in 0..5 {
        let x = det.landmarks[i * 2];
        let y = det.landmarks[i * 2 + 1];
        println!("  {}: ({:.1}, {:.1})", landmark_names[i], x, y);
    }

    // Sample what's actually at those positions in the original image
    let rgb_img = img.to_rgb8();
    println!("\nPixel values at detected landmark positions:");
    for i in 0..5 {
        let x = det.landmarks[i * 2] as u32;
        let y = det.landmarks[i * 2 + 1] as u32;
        if x < width && y < height {
            let pixel = rgb_img.get_pixel(x, y);
            println!(
                "  {}: RGB({}, {}, {})",
                landmark_names[i], pixel[0], pixel[1], pixel[2]
            );
        }
    }

    // Now check the crop
    let cropped = face::align_face(&img, det, 112)?;
    let crop_rgb = cropped.to_rgb8();

    println!("\nTarget positions in 112x112 crop:");
    let ref_landmarks = [
        (38.29, 51.70), // left eye
        (73.53, 51.50), // right eye
        (56.03, 71.74), // nose
        (41.55, 92.37), // left mouth
        (70.73, 92.20), // right mouth
    ];

    println!("\nPixel values at target landmark positions in crop:");
    for i in 0..5 {
        let (x, y) = ref_landmarks[i];
        let x_u = x as u32;
        let y_u = y as u32;
        let pixel = crop_rgb.get_pixel(x_u, y_u);
        println!(
            "  {} at ({:.1}, {:.1}): RGB({}, {}, {})",
            landmark_names[i], x, y, pixel[0], pixel[1], pixel[2]
        );
    }

    // The pixels at eye positions in the crop should match the pixels at eye positions in original
    // If they don't, the transform is mapping the wrong region

    println!("\n⚠️  If eye positions in crop are dark but original eye positions are bright,");
    println!("    the transform is mapping the wrong region!");

    Ok(())
}
