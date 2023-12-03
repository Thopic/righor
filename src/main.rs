mod utils;
use utils::Dna;

fn main() {
    println!("Hello, world!");
    let s: Dna = Dna::from_string("ATCGTCTA");
    println!("{}", Dna::to_string(&s));

}

//	let rng = SmallRng::seed_from_u64(seed);
