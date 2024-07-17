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
use righor::shared::ModelGen;
use righor::Dna;
use righor::EntrySequence;
use std::fs::File;

use righor::shared::model::ModelStructure;
use righor::shared::{
    errors::ErrorConstantRate, errors::ErrorUniformRate, DnaLike, ErrorParameters,
};
use righor::Modelable;

use std::io::{self, BufRead, BufReader};
use std::path::Path;

fn main() -> Result<()> {
    // let mut igor_model = righor::vdj::Model::load_from_files(
    //     Path::new("../Berk_round_15/model_params.txt"),
    //     Path::new("../Berk_round_15/model_marginals.txt"),
    //     Path::new("../Berk_round_15/V_gene_CDR3_anchors.csv"),
    //     Path::new("../Berk_round_15/J_gene_CDR3_anchors.csv"),
    // )?;

    //TODO: modify before release
    let mut igor_model = righor::Model::load_from_name(
        "human",
        "trb",
        None,
        Path::new("/home/thomas/Downloads/righor-py/righor.data/data/righor_models/"),
    )?;
    igor_model.set_model_type(ModelStructure::VxDJ)?;
    igor_model.set_error(ErrorParameters::UniformRate(ErrorUniformRate::default()))?;

    // let sequence = righor::Dna::from_string("GACGCGGAATTCACCCCAAGTCCCACACACCTGATCAAAAAGAGAGCCCAGCAGCTGACTCTGAGATGCTCTCCTAAATCTGAGCATGACAGTGTGTCCTGGTGCCAACAAGCCCTGTGTCAGGGGCCCCAGTTTAACTTTCAGTATTATGAGGAGGAAGAGATTCATAGAGGCAACTACCCTGAACATTTCTCAGGTCCCCAGTTCCTGAACTATAGCTCTGGGCTGAATGTGAACGACCTGTTGCGGTGGGATTCGGCCCTCTATCACTGTGCGAGCAGCAATGACTAGCGAGACCAGTACTTCGGGCCAAGCACGCGACTCCTGGTGCTCG")?;

    let sequence = righor::Dna::from_string("TGTGCCAGCCGACGGACAGCTAACTATGGCTACACCTTC")?;

    let mut align_params = righor::AlignmentParameters::default();
    align_params.left_v_cutoff = 500;
    let mut inference_params = righor::InferenceParameters::default();
    inference_params.min_likelihood = 0.;
    inference_params.min_ratio_likelihood = 0.;
    // let al = igor_model.align_from_cdr3(
    //     &DnaLike::from_dna(sequence),
    //     &igor_model.get_v_segments(),
    //     &igor_model.get_j_segments(),
    // )?;
    let result = if let righor::Model::VDJ(m) = igor_model {
        m.evaluate(
            EntrySequence::NucleotideCDR3((
                DnaLike::from_dna(sequence),
                m.get_v_segments(),
                m.get_j_segments(),
            )),
            &align_params,
            &inference_params,
        )
    } else {
        panic!("")
    };
    println!("{:?}", result);

    // let mut generator = righor::Generator::new(igor_model.clone(), Some(42), None, None)?;

    // let v = (0..10)
    //     .map(|_x| {
    //         righor::EntrySequence::NucleotideSequence(DnaLike::from_dna(
    //             Dna::from_string(&generator.generate_without_errors(true).full_seq).unwrap(),
    //         ))
    //     })
    //     .collect::<Vec<_>>();

    // let mut model = igor_model.uniform()?.clone();

    // println!("Start inference");
    // let mut features = model.infer(&v, None, &align_params, &inference_params)?;

    // for _ii in 1..10 {
    //     println!("{:?}", model.get_error());
    //     features = model.infer(&v, Some(features), &align_params, &inference_params)?;
    // }

    // let mut generator = righor::vdj::Generator::new(igor_model.clone(), Some(42), None, None)?;
    // let mut uniform_model = igor_model.uniform()?;
    // uniform_model.error_rate = 0.1;
    // let mut _inference_params_2 = righor::InferenceParameters {
    //     complete_vdj_inference: true,
    //     ..Default::default()
    // };

    // let mut seq = Vec::new();
    // for _ in tqdm!(0..200) {
    //     let s = righor::Dna::from_string(&generator.generate(false).full_seq)?;
    //     seq.push(s)

    //     // let als = uniform_model.align_sequence(&s.clone(), &align_params)?;
    //     // if !(als.v_genes.is_empty() || als.j_genes.is_empty()) {
    //     //     seq.push(als.clone());
    // }
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

    // for _ii in tqdm![0..6] {
    //     igor_model.align_and_infer(&sequences[0..1000], &align_params, &inference_params)?;
    // }

    // // let mut uniform_model2 = igor_model.uniform()?;

    // // for ii in tqdm![0..5] {
    // //     uniform_model2.infer(&seq, &inference_params_2)?;
    // // }
    // // println!("{:?}", uniform_model2.p_ins_vd);
    // // println!("{:?}", uniform_model.p_ins_vd);
    // // println!("{:?}", igor_model.p_ins_vd);
    Ok(())
}
