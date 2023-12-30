#![allow(unused_imports)] //TODO REMOVE
#![allow(dead_code)]
mod sequence;
mod shared;
pub mod vdj;
pub mod vj;

use anyhow::{anyhow, Result};
use kdam::tqdm;
use ndarray::array;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

fn main() -> Result<()> {
    let mut model = vdj::Model::load_from_files(
        Path::new("models/human_T_beta/model_params.txt"),
        Path::new("models/human_T_beta/model_marginals.txt"),
        Path::new("models/human_T_beta/V_gene_CDR3_anchors.csv"),
        Path::new("models/human_T_beta/J_gene_CDR3_anchors.csv"),
    )?;

    //    model = model.uniform()?;
    model.error_rate = 0.;

    let align_params = sequence::AlignmentParameters {
        min_score_v: 80,
        min_score_j: 10,
        max_error_d: 100,
    };
    let inference_params = shared::InferenceParameters {
        min_likelihood: 1e-60,
        min_likelihood_error: 1e-60,
        min_log_likelihood: -50.0,
        nb_best_events: 10,
        evaluate: true,
    };

    let path = Path::new("tests/data/generated_sequences_no_error.txt");
    let file = File::open(&path)?;
    let lines = io::BufReader::new(file).lines();

    let mut seq = Vec::new();
    for line in tqdm!(lines) {
        let l = line?;

        // let l = "CTCCCAGACATCTGTGTACTTCTGTGCCAGCAGTTTACTGCAAGAGGGGGGGGCAACTAATGAAAAACTGTTTTTTGGCAGTGGAACCCAGCTCTCTGTCTTGG".to_string();
        // println!("{}", l);
        let s = sequence::Dna::from_string(l.trim())?;
        let als = model.align_sequence(s.clone(), &align_params)?;

        let best_al_v = als.best_v_alignment().ok_or(anyhow!("Empty Alignment"))?;
        let best_al_j = als.best_j_alignment().ok_or(anyhow!("Empty Alignment"))?;
        seq.push(als);
    }

    println!("{:?}", model.p_ins_vd);
    for _ in 0..100 {
        let mut features = Vec::new();
        for s in tqdm!(seq.clone().iter(), total = seq.len()) {
            features.push(model.infer_features(&s, &inference_params)?);
        }
        let new_features = vdj::Features::average(features)?;
        model.update(&new_features)?;
        println!("hip");
        println!("{:?}", model.p_ins_vd);
    }

    Ok(())
}
