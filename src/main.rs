#![allow(unused_imports)] //TODO REMOVE
#![allow(dead_code)]
#![warn(clippy::large_types_passed_by_value)]

pub mod shared;
pub mod v_dj;
pub mod vdj;
pub mod vj;

use anyhow::{anyhow, Result};
use kdam::tqdm;
use ndarray::array;
use ndarray::Axis;
use righor::shared::DNAMarkovChain;
use righor::shared::ModelGen;
use righor::AlignmentParameters;
use righor::EntrySequence;
use righor::{AminoAcid, Dna, DnaLike, Gene};
use std::fs::File;

use righor::shared::model::ModelStructure;
use righor::shared::{errors::ErrorConstantRate, errors::ErrorUniformRate, ErrorParameters};
use righor::Modelable;

use std::io::{self, BufRead, BufReader};
use std::path::Path;

fn main() -> Result<()> {
    let mut a: DnaLike = AminoAcid::from_string("RVTVVSPLHF").unwrap().into();
    a = a.extract_subsequence(1, a.len());
    let b = Dna::from_string("GGAGCTCCTATAATTCACCCCTCCACTTT").unwrap();
    println!("{}", a.to_dna().get_string());
    println!("{}", b.get_string());
    println!("{}", a.count_differences(&b));

    // let d = DNAMarkovChain::new(
    //     &array![
    //         [0.1, 0.4, 0.25, 0.25],
    //         [0.25, 0.1, 0.4, 0.1],
    //         [0.25, 0.25, 0.1, 0.4],
    //         [0.25, 0.1, 0.25, 0.25]
    //     ],
    //     true,
    // )?;
    // let mut aa = AminoAcid::from_string("CAMAHDACCCA")?;
    // aa = aa.extract_subsequence(2, 28);
    // println!("{:?} {} {}", aa, aa.start, aa.end);

    // println!(
    //     "LIKELIHOOD: {:.2e}",
    //     d.likelihood_aminoacid(&aa, 1).to_matrix()?
    // );
    // return Ok(());

    //TODO: modify before release
    let mut igor_model = righor::Model::load_from_name(
        "human",
        "trb",
        None,
        Path::new("/home/thomas/Downloads/righor-py/righor.data/data/righor_models/"),
    )?;
    igor_model.set_model_type(ModelStructure::VxDJ)?;
    igor_model.set_error(ErrorParameters::ConstantRate(ErrorConstantRate::new(0.01)))?;

    // println!("{}", igor_model.get_markov_coefficients_vd()?);
    // println!("{}", igor_model.get_markov_coefficients_dj()?);
    // return Ok(());
    let mut generator = righor::Generator::new(igor_model.clone(), Some(42), None, None)?;

    for _ in 0..100 {
        let sequence = generator.generate_without_errors(true);
        println!("{:?}", sequence.clone());
        let cdr3_aa = sequence.cdr3_aa.unwrap();
        //let cdr3_nt = sequence.cdr3_nt;
        let vname = sequence.v_gene;
        let jname = sequence.j_gene;
        // let cdr3_aa = "CASRKRRVTGVSPLHF";
        // let vname = igor_model.get_v_segments()[11].name.clone();
        // let jname = igor_model.get_j_segments()[6].name.clone();
        // println!("{:?}", sequence.cdr3_nt);
        // println!("{:?}", cdr3_aa);

        let a = igor_model.evaluate(
            EntrySequence::NucleotideCDR3((
                AminoAcid::from_string(&cdr3_aa).unwrap().into(),
                igor_model
                    .get_v_segments()
                    .into_iter()
                    .filter(|a| a.name == vname)
                    .collect(),
                igor_model
                    .get_j_segments()
                    .into_iter()
                    .filter(|a| a.name == jname)
                    .collect(),
            )),
            &righor::AlignmentParameters::default(),
            &righor::InferenceParameters::default(),
        )?;

        println!("{:?}", a);
    }

    // let dna = Dna::from_string("TGCGCCAGGAACTATGAACAGTATTTT")?;
    // let es = EntrySequence::NucleotideCDR3((
    //     dna.clone().into(),
    //     vec![igor_model.clone().get_v_segments()[49].clone()],
    //     vec![igor_model.clone().get_j_segments()[9].clone()],
    // ));
    // let mut ip = righor::InferenceParameters::default();
    // ip.infer_features = true;

    // igor_model.set_model_type(righor::shared::ModelStructure::VDJ);
    // let result = igor_model.evaluate(es.clone(), &righor::AlignmentParameters::default(), &ip);
    // println!("VDJ pgen(TGCGCCAGGAACTATGAACAGTATTTT) {:?}\n", result);

    // let igor_model_vdj = match igor_model {
    //     righor::Model::VDJ(ref x) => x,
    //     righor::Model::VJ(_) => panic!("NOPE"),
    // };

    // let result = igor_model_vdj.evaluate_brute_force(
    //     es.clone(),
    //     &righor::AlignmentParameters::default(),
    //     &ip,
    // );
    // println!("VDJ pgen(TGCGCCAGGAACTATGAACAGTATTTT) {:?}\n", result);

    // igor_model.set_model_type(righor::shared::ModelStructure::VxDJ);
    // let result = igor_model.evaluate(es.clone(), &righor::AlignmentParameters::default(), &ip);
    // println!("VxDJ pgen(TGCGCCAGGAACTATGAACAGTATTTT) {:?}\n", result);

    // // let sequence = righor::Dna::from_string("GACGCGGAATTCACCCCAAGTCCCACACACCTGATCAAAAAGAGAGCCCAGCAGCTGACTCTGAGATGCTCTCCTAAATCTGAGCATGACAGTGTGTCCTGGTGCCAACAAGCCCTGTGTCAGGGGCCCCAGTTTAACTTTCAGTATTATGAGGAGGAAGAGATTCATAGAGGCAACTACCCTGAACATTTCTCAGGTCCCCAGTTCCTGAACTATAGCTCTGGGCTGAATGTGAACGACCTGTTGCGGTGGGATTCGGCCCTCTATCACTGTGCGAGCAGCAATGACTAGCGAGACCAGTACTTCGGGCCAAGCACGCGACTCCTGGTGCTCG")?;

    // let mut align_params = righor::AlignmentParameters::default();
    // align_params.left_v_cutoff = 500;
    // let mut inference_params = righor::InferenceParameters::default();
    // inference_params.min_likelihood = 0.;
    // inference_params.min_ratio_likelihood = 0.;
    // inference_params.infer_features = false;

    // let sequence = righor::AminoAcid::from_string("CDF")?;

    // let result = if let righor::Model::VDJ(ref m) = igor_model {
    //     m.evaluate(
    //         EntrySequence::NucleotideCDR3((
    //             DnaLike::from_amino_acid(sequence),
    //             m.get_v_segments()
    //                 .into_iter()
    //                 .filter(|a| a.name == "TRBV9*01")
    //                 .collect(),
    //             m.get_j_segments()
    //                 .into_iter()
    //                 .filter(|a| a.name == "TRBJ2-7*01")
    //                 .collect(),
    //         )),
    //         &align_params,
    //         &inference_params,
    //     )
    // } else {
    //     panic!("")
    // };
    // println!("{:?}", result.unwrap().likelihood);
    // println!("\n");

    // let sequence = righor::Dna::from_string("TGTGACTTC")?;
    // let result = if let righor::Model::VDJ(m) = igor_model {
    //     m.evaluate(
    //         EntrySequence::NucleotideCDR3((
    //             DnaLike::from_dna(sequence),
    //             m.get_v_segments()
    //                 .into_iter()
    //                 .filter(|a| a.name == "TRBV9*01")
    //                 .collect(),
    //             m.get_j_segments()
    //                 .into_iter()
    //                 .filter(|a| a.name == "TRBJ2-7*01")
    //                 .collect(),
    //         )),
    //         &align_params,
    //         &inference_params,
    //     )
    // } else {
    //     panic!("")
    // };
    // println!("{:?}", result.unwrap().likelihood);

    // let sequence = righor::Dna::from_string("TGTGCCAGCCGACGGACAGCTAACTATGGCTACACCTTC")?;

    // let al = igor_model.align_from_cdr3(
    //     &DnaLike::from_dna(sequence),
    //     &igor_model.get_v_segments(),
    //     &igor_model.get_j_segments(),
    // )?;
    // let result = if let righor::Model::VDJ(m) = igor_model {
    //     m.evaluate(
    //         EntrySequence::NucleotideCDR3((
    //             DnaLike::from_dna(sequence),
    //             m.get_v_segments(),
    //             m.get_j_segments(),
    //         )),
    //         &align_params,
    //         &inference_params,
    //     )
    // } else {
    //     panic!("")
    // };
    // println!("{:?}", result);

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
