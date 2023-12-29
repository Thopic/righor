#![allow(unused_imports)] //TODO REMOVE
#![allow(dead_code)]
mod sequence;
mod shared;
pub mod vdj;
pub mod vj;

use anyhow::{anyhow, Result};
use kdam::tqdm;
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

    //model = model.uniform()?;
    model.error_rate = 0.;

    let align_params = sequence::AlignmentParameters {
        min_score_v: 80,
        min_score_j: 10,
        max_error_d: 20,
    };
    let inference_params = shared::InferenceParameters {
        min_likelihood: 1e-60,
        min_likelihood_error: 1e-60,
    };

    let path = Path::new("demo/longer.txt");
    let file = File::open(&path)?;
    let lines = io::BufReader::new(file).lines();

    let mut seq = Vec::new();
    for line in tqdm!(lines) {
        let l = line?;
        println!("{}", l);
        let s = sequence::Dna::from_string(l.trim())?;
        let als = model.align_sequence(s.clone(), &align_params)?;

        let best_al = als.best_v_alignment().ok_or(anyhow!("Empty Alignment"))?;

        //        println!("{}", &als.d_genes.len());
        seq.push(als);

        println!(
            "{} \n {}",
            best_al.index,
            vdj::display_v_alignment(&s, &best_al, &model, &align_params)
        );
        break;
    }

    println!(
        "{:?}",
        model.most_likely_recombinations(&seq[0], 1, &inference_params)
    );

    // for _ in 0..10 {
    //     let mut features = Vec::new();
    //     for s in tqdm!(seq.clone().iter(), total = seq.len()) {
    //         features.push(model.infer_features(&s, &inference_params)?);
    //     }
    //     let new_features = vdj::Features::average(features)?;
    //     model.update(&new_features);
    //     println!("{:?}", model.p_v);
    // }

    Ok(())
}
