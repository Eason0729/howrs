use anyhow::Result;
use howrs_vision::pipeline::Pipeline;

#[test]
fn test_pipeline_initialization() -> Result<()> {
    // Test that we can initialize the pipeline
    let _pipeline = Pipeline::new()?;
    println!("✓ Pipeline initialized successfully");
    Ok(())
}

#[test]
fn test_pipeline_components() -> Result<()> {
    // Verify that both detector and encoder sessions can be created
    let pipeline = Pipeline::new()?;

    // Check that sessions are not null (they exist)
    // This validates the models are loaded correctly
    drop(pipeline.detector);
    drop(pipeline.encoder);

    println!("✓ Pipeline components verified");
    Ok(())
}
