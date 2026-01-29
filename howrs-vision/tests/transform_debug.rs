use anyhow::Result;
use howrs_vision::{face, model};

/// Debug the alignment transform to see what's happening
#[test]
fn test_debug_alignment_transform() -> Result<()> {
    let mut detector = model::detector_session()?;

    // Test both an eason image (broken crop) and ling image (working crop)
    let test_cases = [
        ("test_faces/ir-cam/eason1.png", "eason1"),
        ("test_faces/ir-cam/ling1.png", "ling1"),
    ];

    for (img_path, name) in &test_cases {
        println!("\n=== Debugging {} ===", name);
        let img = image::open(img_path)?;
        let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;

        if detections.is_empty() {
            println!("  No face detected!");
            continue;
        }

        let det = &detections[0];
        println!("  Detection landmarks (source points):");
        for i in 0..5 {
            let x = det.landmarks[i * 2];
            let y = det.landmarks[i * 2 + 1];
            let label = match i {
                0 => "left eye",
                1 => "right eye",
                2 => "nose",
                3 => "left mouth",
                4 => "right mouth",
                _ => "unknown",
            };
            println!("    {}: ({:.2}, {:.2})", label, x, y);
        }

        // Reference landmarks for 112x112
        let ref_landmarks = [
            [38.2946, 51.6963], // left eye
            [73.5318, 51.5014], // right eye
            [56.0252, 71.7366], // nose
            [41.5493, 92.3655], // left mouth
            [70.7299, 92.2041], // right mouth
        ];

        println!("\n  Target reference landmarks (112x112):");
        for (i, [x, y]) in ref_landmarks.iter().enumerate() {
            let label = match i {
                0 => "left eye",
                1 => "right eye",
                2 => "nose",
                3 => "left mouth",
                4 => "right mouth",
                _ => "unknown",
            };
            println!("    {}: ({:.2}, {:.2})", label, x, y);
        }

        // Now test the cropped result
        let cropped = face::align_face(&img, det, 112)?;
        let rgb_img = cropped.to_rgb8();

        // Sample some pixels to see what we got
        println!("\n  Cropped image pixel sampling:");
        println!("    Top-left (0,0): {:?}", rgb_img.get_pixel(0, 0));
        println!("    Top-center (56,0): {:?}", rgb_img.get_pixel(56, 0));
        println!("    Center (56,56): {:?}", rgb_img.get_pixel(56, 56));
        println!("    Eye position (38,51): {:?}", rgb_img.get_pixel(38, 51));
        println!("    Eye position (73,51): {:?}", rgb_img.get_pixel(73, 51));

        // Check brightness in different regions
        let mut top_third: u64 = 0;
        let mut middle_third: u64 = 0;
        let mut bottom_third: u64 = 0;

        for y in 0..112 {
            for x in 0..112 {
                let p = rgb_img.get_pixel(x, y);
                let brightness = p[0] as u64 + p[1] as u64 + p[2] as u64;
                if y < 37 {
                    top_third += brightness;
                } else if y < 74 {
                    middle_third += brightness;
                } else {
                    bottom_third += brightness;
                }
            }
        }

        let pixels_per_third = (112 * 37 * 3) as u64;
        println!("\n  Regional brightness (avg per pixel):");
        println!("    Top third: {}", top_third / pixels_per_third);
        println!("    Middle third: {}", middle_third / pixels_per_third);
        println!("    Bottom third: {}", bottom_third / pixels_per_third);
        println!("    (Eyes should be in top third around y=51, nose middle around y=71)");
    }

    Ok(())
}
