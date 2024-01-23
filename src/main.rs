#![allow(unused_imports)] //TODO REMOVE
#![allow(dead_code)]

mod sequence;
mod shared;
pub mod vdj;
//pub mod vj;

use anyhow::{anyhow, Result};
use kdam::tqdm;
use ndarray::array;
use ndarray::Axis;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

fn main() -> Result<()> {
    let mut igor_model = ihor::vdj::Model::load_from_files(
        Path::new("demo/models/human/tcr_beta/models/model_parms.txt"),
        Path::new("demo/models/human/tcr_beta/models/model_marginals.txt"),
        Path::new("demo/models/human/tcr_beta/ref_genome/V_gene_CDR3_anchors.csv"),
        Path::new("demo/models/human/tcr_beta/ref_genome/J_gene_CDR3_anchors.csv"),
    )?;
    igor_model.error_rate = 0.;

    let align_params = ihor::AlignmentParameters::default();
    let inference_params = ihor::InferenceParameters::default();

    let path = Path::new("../igor_1-4-0/demo/murugan_naive1_noncoding_demo_seqs.txt");
    let file = File::open(&path)?;
    let lines = io::BufReader::new(file).lines();

    let mut seq = Vec::new();
    for line in tqdm!(lines) {
        let l = line?;
        let s = ihor::Dna::from_string(l.trim())?;
        let als = igor_model.align_sequence(s.clone(), &align_params)?;
        if !(als.v_genes.is_empty() || als.j_genes.is_empty()) {
            seq.push(als);
        }
    }
    for _ in 0..1 {
        for s in tqdm!(seq.clone().iter(), total = seq.len()) {
            let result = igor_model.infer(&s, &inference_params).unwrap();
            println!("{:?}", result);
        }
    }

    Ok(())
}
