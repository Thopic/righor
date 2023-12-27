#![allow(unused_imports)] //TODO REMOVE
#![allow(dead_code)]
mod sequence;
mod shared;
pub mod vdj;
pub mod vj;

use anyhow::Result;
use kdam::tqdm;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

fn main() -> Result<()> {
    let mut model = vdj::Model::load_model(
        Path::new("models/human_T_beta/model_params.txt"),
        Path::new("models/human_T_beta/model_marginals.txt"),
        Path::new("models/human_T_beta/V_gene_CDR3_anchors.csv"),
        Path::new("models/human_T_beta/J_gene_CDR3_anchors.csv"),
    )?;
    let align_params = sequence::AlignmentParameters {
        min_score_v: 80,
        min_score_j: 40,
        max_error_d: 5,
    };
    let inference_params = shared::InferenceParameters {
        min_likelihood: 1e-40,
        min_likelihood_error: 1e-60,
    };

    let path = Path::new("demo/murugan_naive1_noncoding_demo_seqs.txt");
    let file = File::open(&path)?;
    let lines = io::BufReader::new(file).lines();

    let mut seq = Vec::new();
    for line in tqdm!(lines) {
        let l = line?;
        seq.push(model.align_sequence(sequence::Dna::from_string(l.trim())?, &align_params)?);
    }

    for _ in 0..10 {
        let mut features = Vec::new();
        for s in tqdm!(seq.clone().iter(), total = seq.len()) {
            features.push(model.infer_features(&s, &inference_params)?);
        }
        let new_features = vdj::Features::average(features)?;
        model.update(&new_features);
    }
    println!("{:?}", model);

    Ok(())
}
