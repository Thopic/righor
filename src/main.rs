#![allow(unused_imports)] //TODO REMOVE
#![allow(dead_code)]
mod model;
mod parser;
mod sequence;
mod utils;
mod utils_sequences;

use indicatif::{ProgressBar, ProgressState, ProgressStyle};
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
        Path::new("models/human_T_beta/model_params.txt"),
        Path::new("models/human_T_beta/model_marginals.txt"),
        Path::new("models/human_T_beta/V_gene_CDR3_anchors.csv"),
        Path::new("models/human_T_beta/J_gene_CDR3_anchors.csv"),
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
        min_score_v: -50,
        min_score_j: -50,
        max_error_v: 10,
        max_error_j: 10,
        max_error_d: 10,
    };

    // Specify the file path
    let path = Path::new("demo/murugan_naive1_noncoding_demo_seqs.txt");

    // Open the file
    let file = File::open(&path)?;
    let file_size = file.metadata()?.len();

    // Create a progress bar
    let pb = ProgressBar::new(file_size);
    pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
		 .unwrap()
		 .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
        .progress_chars("#>-"));

    // Create a buffered reader
    let reader = io::BufReader::new(file);

    // Iterate over each line
    for line in reader.lines() {
        match line {
            Ok(ln) => {
                SequenceVDJ::align_sequence(Dna::from_string(&ln), &model, &align_params);
                pb.inc(ln.len() as u64);
            }
            Err(e) => println!("Error reading line: {}", e),
        }
    }
    pb.finish_with_message("Reading complete");
    Ok(())
    // let seq = SequenceVDJ::align_sequence(nt_seq, &model, &align_params);
    // // println!("{:?}", seq.v_genes);
}
