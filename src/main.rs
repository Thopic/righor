#![allow(unused_imports)] //TODO REMOVE
#![allow(dead_code)]

mod sequence;
mod shared;
pub mod vdj;
pub mod vj;

use anyhow::{anyhow, Result};
use kdam::tqdm;
use ndarray::array;
use ndarray::Axis;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

fn main() -> Result<()> {
    // let mut igor_model = righor::vdj::Model::load_from_files(
    //     Path::new("models/human_T_beta/model_params.txt"),
    //     Path::new("models/human_T_beta/model_marginals.txt"),
    //     Path::new("models/human_T_beta/V_gene_CDR3_anchors.csv"),
    //     Path::new("models/human_T_beta/J_gene_CDR3_anchors.csv"),
    // )?;

    //TODO: modify before release
    let mut igor_model = righor::vj::Model::load_from_name(
        "human",
        "tra",
        None,
        Path::new("/home/thomas/Downloads/righor-py/models"),
    )?;

    igor_model.save_model(Path::new("tmp/"));

    let mut new_model = righor::vj::Model::load_from_files(
        Path::new("tmp/model_params.txt"),
        Path::new("tmp/model_marginals.txt"),
        Path::new("tmp/V_gene_CDR3_anchors.csv"),
        Path::new("tmp/J_gene_CDR3_anchors.csv"),
    )?;

    println!("{}", igor_model.similar_to(new_model));

    // igor_model.error_rate = 0.;

    // let mut generator = righor::vdj::Generator::new(igor_model.clone(), Some(42), None, None)?;
    // let mut uniform_model = igor_model.uniform()?;
    // let align_params = righor::AlignmentParameters::default();
    // let inference_params = righor::InferenceParameters::default();

    // let mut seq = Vec::new();
    // for _ in tqdm!(0..1) {
    //     let s = righor::Dna::from_string(&generator.generate(false).full_seq)?;
    //     let als = uniform_model.align_sequence(s.clone(), &align_params)?;
    //     if !(als.v_genes.is_empty() || als.j_genes.is_empty()) {
    //         seq.push(als);
    //     }
    // }
    // for ii in 0..5 {
    //     let _ = uniform_model.infer(&seq, &inference_params);
    //     println!("{:?}", ii);
    // }

    // println!("{:?}", uniform_model.p_ins_vd);

    Ok(())
}
