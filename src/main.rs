#![allow(unused_imports)] //TODO REMOVE
#![allow(dead_code)]

mod sequence;
mod shared;
pub mod vdj;
//pub mod vj;

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

    model = model.uniform()?;
    //    model.error_rate = 0.;

    let align_params = sequence::AlignmentParameters {
        min_score_v: 20,
        min_score_j: 20,
        max_error_d: 100,
    };
    let inference_params = shared::InferenceParameters {
        min_log_likelihood: -400.0,
        nb_best_events: 10,
        evaluate: true,
    };

    let path = Path::new("demo/murugan_naive1_noncoding_demo_seqs.txt");
    let file = File::open(&path)?;
    let lines = io::BufReader::new(file).lines();

    let mut seq = Vec::new();
    for line in tqdm!(lines) {
        let l = line?;

        //let l = "CTCCCAGACATCTGTGTACTTCTGTGCCAGCAGTTTACTGCAAGAGGGGGGGGCAACTAATGAAAAACTGTTTTTTGGCAGTGGAACCCAGCTCTCTGTCTTGG".to_string();
        //println!("{}", l);
        let s = sequence::Dna::from_string(l.trim())?;
        let als = model.align_sequence(s.clone(), &align_params)?;
        if !(als.v_genes.is_empty() || als.j_genes.is_empty()) {
            seq.push(als);
        }
        //        break;
    }

    println!("{:?}", model.p_ins_vd);
    for _ in 0..5 {
        let mut features = Vec::new();
        for s in tqdm!(seq.clone().iter(), total = seq.len()) {
            // println!(
            //     "{:?}",
            //     model.most_likely_recombinations(&s, 10, &inference_params)
            // );
            features.push(model.infer_features(&s, &inference_params)?);
        }
        println!("{:?}", features[0].insvd.length_distribution);
        let new_features = vdj::Features::average(features)?;
        model.update(&new_features)?;
        println!("hip");
        println!("{:?}", model.p_ins_vd);
    }

    Ok(())
}
