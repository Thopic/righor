use anyhow::Result;
use righor;
use ndarray::array;
use std::path::Path;

pub mod common;

#[test]
fn test_generation_real() -> Result<()> {
    // Basic test, ne inference with a real model and a real sequence
    // Just check that nothing panic or return an error.
    // Note: this rely on the presence of data files, so it may
    // fail if the data files are not present
    let model = righor::vdj::Model::load_from_files(
        Path::new("models/human_T_beta/model_params.txt"),
        Path::new("models/human_T_beta/model_marginals.txt"),
        Path::new("models/human_T_beta/V_gene_CDR3_anchors.csv"),
        Path::new("models/human_T_beta/J_gene_CDR3_anchors.csv"),
    )?;
    let mut gen = righor::vdj::Generator::new(model, Some(0));
    for _ in 0..100 {
        println!("{}", gen.generate(true).full_seq);
    }
    Ok(())
}

#[test]
fn test_generation_simple_model() -> Result<()> {
    // Basic test, ne inference with a real model and a real sequence
    // Just check that nothing panic or return an error.
    // Note: this rely on the presence of data files, so it may
    // fail if the data files are not present
    let model = common::simple_model_vdj();
    let mut gen = righor::vdj::Generator::new(model, Some(0));
    for _ in 0..100 {
        println!("{}", gen.generate(true).full_seq);
    }
    Ok(())
}
