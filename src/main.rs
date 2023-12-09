mod utils;
mod parser;
mod model;
use parser::{ParserMarginals, ParserParams};
use std::path::Path;
use model::{ModelVDJ, ModelVJ};
use rand::{SeedableRng};
use rand::rngs::{SmallRng};
use utils::{Dna};


fn main() {
    // let mut model = ModelVDJ::load_model(Path::new("models/human_T_beta/model_params.txt"),
    // 					 Path::new("models/human_T_beta/model_marginals.txt"),
    // 					 Path::new("models/human_T_beta/V_gene_CDR3_anchors.csv"),
    // 					 Path::new("models/human_T_beta/J_gene_CDR3_anchors.csv")).unwrap();

    // let mut small_rng = SmallRng::seed_from_u64(42);
    // let (dna, _, v, j) = model.generate(true, &mut small_rng);
    // println!("{:?}", model.recreate_full_sequence(dna, v, j).0.to_string());

    // let mut model = ModelVJ::load_model(Path::new("models/human_T_alpha/model_params.txt"),
    // 					 Path::new("models/human_T_alpha/model_marginals.txt"),
    // 					 Path::new("models/human_T_alpha/V_gene_CDR3_anchors.csv"),
    // 					 Path::new("models/human_T_alpha/J_gene_CDR3_anchors.csv")).unwrap();


    // let (dna, aa, v, j) = model.generate(true, &mut small_rng);
    // println!("{:?}", model.recreate_full_sequence(dna, v, j).0.to_string());

    let x = Dna::from_string("ATACGATCATTGACAATCTAGAGATAC");
    let y = Dna::from_string("CAATCTAGAGATTCAGACATGAAACCAAA");
    let alignment = Dna::align_left_right(&x, &y);
    println!("{}", alignment.pretty(x.seq.as_slice(), y.seq.as_slice(), 80));
}
