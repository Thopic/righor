mod utils;
mod parser;
use utils::Dna;
use parser::ParserMarginals;

fn main() {
    println!("Hello, world!");
    let s: Dna = Dna::from_string("ATCGTCTA");
    println!("{}", Dna::to_string(&s));

    let pm: ParserMarginals;
    pm.parse("models/model_marginals.txt")

}

//	let rng = SmallRng::seed_from_u64(seed);
