use anyhow::Result;
use ihor;
use ndarray::array;
use std::path::Path;

#[test]
fn test_infer_feature_real() -> Result<()> {
    // Basic test, ne inference with a real model and a real sequence
    // Just check that nothing panic or return an error.
    // Note: this rely on the presence of data files, so it may
    // fail if the data files are not present
    let mut model = ihor::vdj::Model::load_from_files(
        Path::new("models/human_T_beta/model_params.txt"),
        Path::new("models/human_T_beta/model_marginals.txt"),
        Path::new("models/human_T_beta/V_gene_CDR3_anchors.csv"),
        Path::new("models/human_T_beta/J_gene_CDR3_anchors.csv"),
    )?;
    let mut gen = ihor::vdj::Generator::new(model, Some(0));
    for _ in 0..100 {
        println!("{}", gen.generate(true).full_seq);
    }
    Ok(())
}
