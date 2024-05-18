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
use righor::EntrySequence;
use std::fs::File;

use righor::shared::model::ModelStructure;
use righor::shared::{errors::ErrorUniformRate, ErrorParameters};
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
        "igh",
        None,
        Path::new("/home/thomas/Downloads/righor-py/righor.data/data/righor_models/"),
    )?;
    igor_model.set_model_type(ModelStructure::VxDJ)?;
    igor_model.set_error(ErrorParameters::UniformRate(ErrorUniformRate::default()))?;

    // let sequence = righor::Dna::from_string("GACGCGGAATTCACCCCAAGTCCCACACACCTGATCAAAAAGAGAGCCCAGCAGCTGACTCTGAGATGCTCTCCTAAATCTGAGCATGACAGTGTGTCCTGGTGCCAACAAGCCCTGTGTCAGGGGCCCCAGTTTAACTTTCAGTATTATGAGGAGGAAGAGATTCATAGAGGCAACTACCCTGAACATTTCTCAGGTCCCCAGTTCCTGAACTATAGCTCTGGGCTGAATGTGAACGACCTGTTGCGGTGGGATTCGGCCCTCTATCACTGTGCGAGCAGCAATGACTAGCGAGACCAGTACTTCGGGCCAAGCACGCGACTCCTGGTGCTCG")?;

    let sequence = righor::Dna::from_string("ACCCTCCAGTCTGCCAGGCCCTCACATACCTCTCAGTACCTCTGTGCCAGCAGTGAGGACAGGGACGTCACTGAAGCTTTCTTTGGACAAGGCACC")?;

    let mut align_params = righor::AlignmentParameters::default();
    align_params.left_v_cutoff = 500;
    let mut inference_params = righor::InferenceParameters::default();
    inference_params.min_likelihood = 0.;
    inference_params.min_ratio_likelihood = 0.;
    let al = igor_model.align_sequence(&sequence, &align_params)?;
    let _result = igor_model.evaluate(EntrySequence::Aligned(al), &align_params, &inference_params);

    // for v in igor_model.get_v_segments() {
    //     let mut gen = righor::vdj::Generator::new(
    //         igor_model.clone(),
    //         Some(42),
    //         Some(vec![v]),
    //         Some(igor_model.clone().get_j_segments()),
    //     )?;
    //     for _ in 0..1000 {
    //         println!("{:?}", gen.generate(false));
    //     }
    // }

    // let file = File::open("test_sequences.txt")?;
    // let reader = BufReader::new(file);
    // let mut sequences: Vec<righor::Dna> = Vec::new();

    // for line in reader.lines() {
    //     let line = line?.trim().to_string();
    //     let dna = righor::Dna::from_string(&line)?;
    //     sequences.push(dna);
    // }
    // println!("SEQUENCES LOADED");
    // // let mut generator = righor::vdj::Generator::new(igor_model.clone(), Some(42), None, None)?;
    // // for _ in 0..100 {
    // //     generator.generate_without_errors(true);
    // // }
    // // igor_model.save_model(Path::new("../tmp/"))?;
    // // igor_model.error_rate = 0.1;

    // // let mut generator = righor::vdj::Generator::new(igor_model.clone(), Some(42), None, None)?;
    // // let mut uniform_model = igor_model.uniform()?;
    // // uniform_model.error_rate = 0.1;
    // // let mut _inference_params_2 = righor::InferenceParameters {
    // //     complete_vdj_inference: true,
    // //     ..Default::default()
    // // };

    // // let mut seq = Vec::new();
    // // for _ in tqdm!(0..200) {
    // //     let s = righor::Dna::from_string(&generator.generate(false).full_seq)?;
    // //     seq.push(s)

    // //     // let als = uniform_model.align_sequence(&s.clone(), &align_params)?;
    // //     // if !(als.v_genes.is_empty() || als.j_genes.is_empty()) {
    // //     //     seq.push(als.clone());
    // // }
    // // println!(
    // //     "{}",
    // //     igor_model
    // //         .evaluate(&als.clone(), &inference_params)?
    // //         .likelihood
    // // );
    // // println!(
    // //     "{}",
    // //     igor_model
    // //         .evaluate(&als.clone(), &inference_params_2)?
    // //         .likelihood
    // // );

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
