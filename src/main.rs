use std::env;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use howrs::{config, identity, matcher, storage, Embedding, Pipeline};
use howrs_vision::video::Camera;
use log::{info, warn};

#[derive(Parser)]
#[command(name = "howrs")]
#[command(
    version,
    about = "Rust howdy clone - facial recognition authentication"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Enroll face from camera
    Enroll {
        /// User ID to enroll (defaults to current user)
        #[arg(short, long)]
        user: Option<String>,
    },
    /// Test authentication by matching against enrolled faces
    Test {
        /// User ID to test (defaults to current user)
        #[arg(short, long)]
        user: Option<String>,
    },
    /// Remove all enrolled faces for a user
    Purge {
        /// User ID to purge (defaults to current user)
        #[arg(short, long)]
        user: Option<String>,
    },
    /// Open config file in editor
    Config,
}

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .format_target(false)
        .format_timestamp(None)
        .init();

    let cli = Cli::parse();
    let cfg = config::load_config(None)?;

    // Determine user ID
    let default_user = match env::var("SUDO_USER") {
        Ok(x) => x,
        Err(_) => identity::current_user_id()?,
    };

    match cli.command {
        Commands::Enroll { user } => {
            let user_id = user.unwrap_or(default_user);
            enroll(&cfg, &user_id)
        }
        Commands::Test { user } => {
            let user_id = user.unwrap_or(default_user);
            test(&cfg, &user_id)
        }
        Commands::Purge { user } => {
            let user_id = user.unwrap_or(default_user);
            purge(&user_id)
        }
        Commands::Config => open_config(),
    }
}

fn enroll(cfg: &config::Config, user_id: &str) -> Result<()> {
    info!("Enrolling user: {}", user_id);
    info!("Opening camera: {}", cfg.camera);

    let mut camera = Camera::open(&cfg.camera).context("Failed to open camera")?;

    let mut pipeline = Pipeline::new().context("Failed to initialize face recognition pipeline")?;

    info!("Camera opened. Capturing frames...");
    info!("Press Ctrl+C to stop.");

    // Capture multiple frames and try to get a good face
    let max_attempts = 30;
    let mut best_detection: Option<howrs::Detection> = None;
    let mut best_embedding: Option<Embedding> = None;

    for i in 0..max_attempts {
        let frame = camera.frame().context("Failed to capture frame")?;

        let img = image::DynamicImage::ImageRgb8(frame);

        match pipeline.process_image(&img, 0.6, 0.3) {
            Ok((detection, embedding)) => {
                info!(
                    "Frame {}: Face detected with score {:.3}",
                    i + 1,
                    detection.score
                );

                // Keep the best detection
                let score = detection.score;
                if best_detection.is_none() || score > best_detection.as_ref().unwrap().score {
                    best_detection = Some(detection);
                    best_embedding = Some(embedding);
                }

                // If we got a good enough detection, we're done
                if score > 0.8 {
                    info!("High quality face detected!");
                    break;
                }
            }
            Err(e) => {
                warn!("Frame {}: {}", i + 1, e);
            }
        }

        // Small delay between frames
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    match (best_detection, best_embedding) {
        (Some(detection), Some(embedding)) => {
            info!("Best face: score {:.3}", detection.score);

            // Save embedding
            let record = storage::FaceRecord {
                id: uuid::Uuid::new_v4().to_string(),
                embedding: embedding.vector.iter().copied().collect(),
            };

            storage::save_record(user_id, record).context("Failed to save face record")?;

            info!("✓ Face enrolled successfully for user: {}", user_id);
            Ok(())
        }
        _ => {
            anyhow::bail!(
                "Failed to detect a face. Please ensure your face is visible and well-lit."
            );
        }
    }
}

fn test(cfg: &config::Config, user_id: &str) -> Result<()> {
    info!("Testing authentication for user: {}", user_id);

    // Load enrolled faces
    let records = storage::load_records(user_id).context("Failed to load face records")?;

    if records.is_empty() {
        anyhow::bail!(
            "No enrolled faces found for user: {}. Run 'enroll' first.",
            user_id
        );
    }

    info!("Found {} enrolled face(s)", records.len());
    info!("Opening camera: {}", cfg.camera);

    let mut camera = Camera::open(&cfg.camera).context("Failed to open camera")?;

    let mut pipeline = Pipeline::new().context("Failed to initialize face recognition pipeline")?;

    info!("Camera opened. Capturing frames...");

    // Try multiple frames
    let max_attempts = 30;

    for i in 0..max_attempts {
        let frame = camera.frame().context("Failed to capture frame")?;

        let img = image::DynamicImage::ImageRgb8(frame);

        match pipeline.extract_embedding(&img, cfg.threshold, 0.3) {
            Ok(probe_embedding) => {
                info!("Frame {}: Face detected", i + 1);

                // Match against stored faces
                let best_score = matcher::best_score(&records, &probe_embedding);

                if let Some(score) = best_score {
                    info!(
                        "Match score: {:.3} (threshold: {:.3})",
                        score, cfg.threshold
                    );

                    if score >= cfg.threshold {
                        info!("✓ Authentication successful!");
                        return Ok(());
                    }
                }
            }
            Err(e) => {
                warn!("Frame {}: {}", i + 1, e);
            }
        }

        // Small delay between frames
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    anyhow::bail!("Authentication failed: No matching face detected")
}

fn purge(user_id: &str) -> Result<()> {
    info!("Purging enrolled faces for user: {}", user_id);

    storage::purge(user_id).context("Failed to purge face records")?;

    info!("✓ All faces purged for user: {}", user_id);
    Ok(())
}

fn open_config() -> Result<()> {
    let config_path = config::CONFIG_PATH.as_os_str();
    let editor = env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());

    info!("Opening config file: {:?}", config_path);

    let status = std::process::Command::new(editor)
        .arg(config_path)
        .status()
        .context("Failed to open editor")?;

    if !status.success() {
        anyhow::bail!("Editor exited with non-zero status");
    }

    Ok(())
}
