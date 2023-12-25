#![allow(unused_imports)] //TODO REMOVE
#![allow(dead_code)]
mod feature;
mod inference;
mod model;
mod parser;
mod sequence;
mod utils;
mod utils_sequences;

use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use inference::{FeaturesVDJ, InferenceParameters};
use model::{ModelVDJ, ModelVJ};
use parser::{ParserMarginals, ParserParams};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use sequence::SequenceVDJ;
use std::fmt::Write;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use utils_sequences::{AlignmentParameters, Dna};

fn main() -> io::Result<()> {
    let model = ModelVDJ::load_model(
        Path::new("tests/models/onegene/model_params.txt"),
        Path::new("tests/models/onegene/model_marginals.txt"),
        Path::new("tests/models/onegene/V_gene_CDR3_anchors.csv"),
        Path::new("tests/models/onegene/J_gene_CDR3_anchors.csv"),
    )
    .unwrap();

    // let mut small_rng = SmallRng::seed_from_u64(42);
    // let (dna, _, v, j) = model.generate(true, &mut small_rng);
    // println!("{:?}", model.recreate_full_sequence(dna, v, j).0.to_string());

    // let mut model = ModelVJ::load_model(Path::new("models/human_T_alpha/model_params.txt"),
    // 					 Path::new("models/human_T_alpha/model_marginals.txt"),
    // 					 Path::new("models/human_T_alpha/V_gene_CDR3_anchors.csv"),
    // 					 Path::new("models/human_T_alpha/J_gene_CDR3_anchors.csv")).unwrap();

    // let (dna, aa, v, j) = model.generate(true, &mut small_rng);
    // println!("{:?}", model.recreate_full_sequence(dna, v, j).0.to_string());

    // let nt_seq = Dna::from_string("TCAGAACCCAGGGACTCAGCTGTGTATTTTTGTGCTAGTGGTTTGGTACAATCAGCCCCA");

    let align_params = AlignmentParameters {
        min_score_v: 40,
        min_score_j: 10,
        max_error_d: 30,
    };

    let infer_params = InferenceParameters {
        //nb_rounds_em: 3,
        min_likelihood: 1e-40,
        min_likelihood_error: 1e-60,
    };

    // Specify the file path
    let path = Path::new("tests/models/seq_onegene.txt");

    // Open the file
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();

    // Create a progress bar
    let pb = ProgressBar::new(file_size);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:7}/{len:7} ({eta})",
        )
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
            write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
        })
        .progress_chars("#>-"),
    );

    // Create a buffered reader
    let reader = io::BufReader::new(file);

    let mut sequences = Vec::<SequenceVDJ>::new();

    // Iterate over each line
    for line in reader.lines() {
        match line {
            Ok(ln) => {
                sequences.push(
                    SequenceVDJ::align_sequence(
                        Dna::from_string(&ln).unwrap(),
                        &model,
                        &align_params,
                    )
                    .unwrap(),
                );
                pb.inc(ln.len() as u64);
            }
            Err(e) => println!("Error reading line: {e}"),
        }
    }
    pb.finish_with_message("Reading complete");
    println!("{:?}", sequences[0]);

    let margs = FeaturesVDJ::new(&model, &infer_params).unwrap();
    //    println!("{:?}", margs);
    println!("Expectation-Maximization");
    //    let margs_new = expectation_maximization(&margs, &sequences, &infer_params);
    //println!("{:?}", margs_new.unwrap());
    // println!("{:?}", margs);
    Ok(())
}
