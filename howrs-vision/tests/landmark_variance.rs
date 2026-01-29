/// Test different variance values for landmark decoding
/// According to RetinaFace and similar face detectors, landmarks often use variance [0.1, 0.2]
/// for bbox but might use different values (or no variance) for landmarks
use anyhow::Result;

#[test]
#[ignore]
fn test_different_landmark_variances() -> Result<()> {
    // From debug_landmark_decoding test, we have:
    // Prior: cx=0.4750, cy=0.5750, w=0.2000, h=0.2000
    // Left eye delta: [0.3909, -0.6648]
    // Right eye delta: [1.9756, -0.4757]
    // bbox width: 163.9 px

    let prior = [0.4750, 0.5750, 0.2000, 0.2000];
    let left_eye_delta = [0.3909, -0.6648];
    let right_eye_delta = [1.9756, -0.4757];
    let bbox_width_px = 163.9;

    println!("Testing different variance/scaling approaches:");
    println!("================================================\n");

    // Current implementation: variance[0] = 0.1
    println!("1. Current (variance[0] = 0.1):");
    test_variance(
        &prior,
        &left_eye_delta,
        &right_eye_delta,
        0.1,
        bbox_width_px,
    );

    // Try no variance (direct addition)
    println!("\n2. No variance (multiply delta by 1.0):");
    test_variance(
        &prior,
        &left_eye_delta,
        &right_eye_delta,
        1.0,
        bbox_width_px,
    );

    // Try variance[1] = 0.2
    println!("\n3. Using variance[1] = 0.2:");
    test_variance(
        &prior,
        &left_eye_delta,
        &right_eye_delta,
        0.2,
        bbox_width_px,
    );

    // Try half variance
    println!("\n4. Half variance (0.05):");
    test_variance(
        &prior,
        &left_eye_delta,
        &right_eye_delta,
        0.05,
        bbox_width_px,
    );

    // Try double variance
    println!("\n5. Double variance (0.2) - same as variance[1]:");
    test_variance(
        &prior,
        &left_eye_delta,
        &right_eye_delta,
        0.2,
        bbox_width_px,
    );

    // Try 0.25
    println!("\n6. Variance = 0.25:");
    test_variance(
        &prior,
        &left_eye_delta,
        &right_eye_delta,
        0.25,
        bbox_width_px,
    );

    // Try 0.3
    println!("\n7. Variance = 0.3:");
    test_variance(
        &prior,
        &left_eye_delta,
        &right_eye_delta,
        0.3,
        bbox_width_px,
    );

    println!("\n================================================");
    println!("Expected: Eye distance should be ~30-35% of face width");
    println!(
        "Target: {} - {} px",
        bbox_width_px * 0.30,
        bbox_width_px * 0.35
    );

    Ok(())
}

fn test_variance(
    prior: &[f32; 4],
    left_delta: &[f32; 2],
    right_delta: &[f32; 2],
    variance: f32,
    bbox_width_px: f32,
) {
    let left_x = prior[0] + left_delta[0] * variance * prior[2];
    let left_y = prior[1] + left_delta[1] * variance * prior[3];
    let right_x = prior[0] + right_delta[0] * variance * prior[2];
    let right_y = prior[1] + right_delta[1] * variance * prior[3];

    let dist_norm = ((right_x - left_x).powi(2) + (right_y - left_y).powi(2)).sqrt();
    let dist_px = dist_norm * 640.0;
    let ratio = (dist_px / bbox_width_px) * 100.0;

    println!("  Left eye: [{:.4}, {:.4}]", left_x, left_y);
    println!("  Right eye: [{:.4}, {:.4}]", right_x, right_y);
    println!(
        "  Distance: {:.1} px ({:.1}% of face width)",
        dist_px, ratio
    );
}
