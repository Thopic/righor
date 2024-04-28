#![allow(unused_imports)] //TODO REMOVE
#![allow(dead_code)]

pub mod shared;
pub mod v_dj;
pub mod vdj;
pub mod vj;

use anyhow::{anyhow, Result};
use kdam::tqdm;
use ndarray::array;
use ndarray::Axis;

use righor::Modelable;
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
    let mut igor_model = righor::vdj::Model::load_from_name(
        "human",
        "trb",
        None,
        Path::new("/home/thomas/Downloads/righor-py/righor.data/data/righor_models/"),
    )?;
    let mut generator = righor::vdj::Generator::new(igor_model.clone(), Some(42), None, None)?;

    for _ in 0..100 {
        generator.generate_without_errors(true);
    }

    igor_model.save_model(Path::new("../tmp/"))?;
    igor_model.error_rate = 0.;

    let mut generator = righor::vdj::Generator::new(igor_model.clone(), Some(42), None, None)?;
    let mut uniform_model = igor_model.uniform()?;
    let align_params = righor::AlignmentParameters::default();
    let inference_params = righor::InferenceParameters::default();
    let mut _inference_params_2 = righor::InferenceParameters {
        complete_vdj_inference: true,
        ..Default::default()
    };

    let mut seq = Vec::new();
    for _ in tqdm!(0..10000) {
        let s = righor::Dna::from_string(&generator.generate(false).full_seq)?;
        seq.push(s)

        // let als = uniform_model.align_sequence(&s.clone(), &align_params)?;
        // if !(als.v_genes.is_empty() || als.j_genes.is_empty()) {
        //     seq.push(als.clone());
    }
    // println!(
    //     "{}",
    //     igor_model
    //         .evaluate(&als.clone(), &inference_params)?
    //         .likelihood
    // );
    // println!(
    //     "{}",
    //     igor_model
    //         .evaluate(&als.clone(), &inference_params_2)?
    //         .likelihood
    // );

    for _ii in tqdm![0..6] {
        uniform_model.align_and_infer(&seq, &align_params, &inference_params)?;
    }

    // let mut uniform_model2 = igor_model.uniform()?;

    // for ii in tqdm![0..5] {
    //     uniform_model2.infer(&seq, &inference_params_2)?;
    // }
    // println!("{:?}", uniform_model2.p_ins_vd);
    // println!("{:?}", uniform_model.p_ins_vd);
    // println!("{:?}", igor_model.p_ins_vd);
    Ok(())
}
