use phf::phf_map;
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use rand_distr::WeightedAliasIndex;
use ndarray_linalg::{Eig};
use ndarray::{Array2, Axis};
use std::error::Error;
use bio::alignment::{pairwise, Alignment};


// Define DNA and AminoAcid structure + basic operations on the sequences

static DNA_TO_AMINO: phf::Map<&'static str, u8> = phf_map! {
    "TTT" => b'F', "TTC" => b'F', "TTA" => b'L', "TTG" => b'L', "TCT" => b'S', "TCC" => b'S',
    "TCA" => b'S', "TCG" => b'S', "TAT" => b'Y', "TAC" => b'Y', "TAA" => b'*', "TAG" => b'*',
    "TGT" => b'C', "TGC" => b'C', "TGA" => b'*', "TGG" => b'W', "CTT" => b'L', "CTC" => b'L',
    "CTA" => b'L', "CTG" => b'L', "CCT" => b'P', "CCC" => b'P', "CCA" => b'P', "CCG" => b'P',
    "CAT" => b'H', "CAC" => b'H', "CAA" => b'Q', "CAG" => b'Q', "CGT" => b'R', "CGC" => b'R',
    "CGA" => b'R', "CGG" => b'R', "ATT" => b'I', "ATC" => b'I', "ATA" => b'I', "ATG" => b'M',
    "ACT" => b'T', "ACC" => b'T', "ACA" => b'T', "ACG" => b'T', "AAT" => b'N', "AAC" => b'N',
    "AAA" => b'K', "AAG" => b'K', "AGT" => b'S', "AGC" => b'S', "AGA" => b'R', "AGG" => b'R',
    "GTT" => b'V', "GTC" => b'V', "GTA" => b'V', "GTG" => b'V', "GCT" => b'A', "GCC" => b'A',
    "GCA" => b'A', "GCG" => b'A', "GAT" => b'D', "GAC" => b'D', "GAA" => b'E', "GAG" => b'E',
    "GGT" => b'G', "GGC" => b'G', "GGA" => b'G', "GGG" => b'G'
};

// The standard ACGT nucleotides
const NUCLEOTIDES: [u8; 4] = [b'A', b'C', b'G', b'T'];

static COMPLEMENT: phf::Map<u8, u8> = phf_map! {
    b'A' => b'T', b'T' => b'A', b'G' => b'C', b'C' => b'G'
};





#[derive(Default, Clone, Debug)]
pub struct Dna{
    pub seq: Vec<u8>,
}

#[derive(Default, Clone, Debug)]
pub struct AminoAcid{
    pub seq: Vec<u8>,
}


impl Dna{

    pub fn new() -> Dna {
	Dna{seq:Vec::new()}
    }

    pub fn from_string(s: &str) -> Dna {
	return Dna{seq: s.as_bytes().to_vec()};
    }

    pub fn to_string(&self) -> String {
        return String::from_utf8_lossy(&self.seq).to_string();
    }

    pub fn translate(&self) -> Result<AminoAcid, &'static str> {
        if self.seq.len() % 3 != 0 {
            return Err("Translation not possible, invalid length.");
        }

        let amino_sequence: Vec<u8> = self.seq
            .chunks(3)
            .filter_map(|codon| {
		let codon_str = std::str::from_utf8(codon).ok()?;
                DNA_TO_AMINO.get(codon_str).copied()
            })
            .collect();
        Ok(AminoAcid{seq:amino_sequence})
    }
    pub fn reverse_complement(&self) -> Dna {
	Dna{seq: self.seq.iter().filter_map(|x| COMPLEMENT.get(x).copied()).rev().collect()}
    }

    pub fn extract_subsequence(&self, start: usize, end:usize) -> Dna{
	Dna{seq: self.seq[start..end].to_vec()}
    }

    pub fn len(&self) -> usize {
	self.seq.len()
    }

    pub fn extend(&mut self, dna: &Dna) {
	self.seq.extend(dna.seq.iter());
    }

    pub fn reverse(&mut self) {
	self.seq.reverse();
    }


    // TODO: make a struct with the alignments parameters
    pub fn align_left_right(sleft: &Dna, sright: &Dna) -> Alignment {
	// Align two sequences with this format
	// ACATCCCACCATTCA
	//         CCATGACTCATGAC

	// these parameters are a bit ad hoc
	let scoring = pairwise::Scoring {
	    gap_open: -30,
	    gap_extend: -2,
	    match_fn: |a: u8, b: u8| if a == b {6i32} else {-6i32}, // TODO: deal better with possible IUPAC codes
	    match_scores: None,
	    xclip_prefix: 0,
	    xclip_suffix: pairwise::MIN_SCORE,
	    yclip_prefix: pairwise::MIN_SCORE,
	    yclip_suffix: 0,
	};

	let mut aligner = pairwise::Aligner::with_capacity_and_scoring(sleft.len(),
							     sright.len(),
							     scoring);

	aligner.custom(sleft.seq.as_slice(), sright.seq.as_slice())
    }



}

impl AminoAcid{
    pub fn from_string(s: &str) -> AminoAcid {
	return AminoAcid{seq: s.as_bytes().to_vec()};
    }
    pub fn to_string(&self) -> String {
        return String::from_utf8_lossy(&self.seq).to_string();
    }
}


// Define some storage wrapper for the V/D/J genes
#[derive(Default, Clone, Debug)]
pub struct Gene {
    pub name: String,
    pub seq: Dna,
    pub functional: String,
    pub cdr3_pos: Option<usize> // start (for V gene) or end (for J gene) of CDR3
}


// Define the random distributions used in the rest of the code.


// Generate an integer with a given probability
#[derive(Clone, Debug)]
pub struct DiscreteDistribution {
    distribution: WeightedAliasIndex<f64>,
}


impl DiscreteDistribution {
    pub fn new(weights: Vec<f64>) -> Result<Self, Box<dyn Error>> {
	if !weights.iter().all(|&x| x >= 0.) {
	    return Err("Error when creating distribution: negative weights")?;
	}

	let distribution = match weights.iter().sum::<f64>().abs() < 1e-10 {
	    true => WeightedAliasIndex::new(vec![1.; weights.len()]) // when all the value are 0, all the values are equiprobable.
		.map_err(|e| format!("Error when creating distribution: {}", e))?,
	    false => WeightedAliasIndex::new(weights)
		.map_err(|e| format!("Error when creating distribution: {}", e))?
	};
        Ok(DiscreteDistribution { distribution:distribution })
    }

    pub fn generate<R: Rng>(&mut self, rng: &mut R) -> usize {
        self.distribution.sample(rng)
    }
}


impl Default for DiscreteDistribution {
    fn default() -> Self {
        return DiscreteDistribution {
	    distribution: WeightedAliasIndex::new(vec![1.]).unwrap()
	};
    }
}


// Markov chain structure (for the insertion process)
#[derive(Default, Clone, Debug)]
pub struct MarkovDNA {
    initial_distribution: DiscreteDistribution, // first nucleotide, ACGT order
    transition_matrix: Vec<DiscreteDistribution>, // Markov matrix, ACGT order
}



impl MarkovDNA {

    pub fn new(transition_probs: Array2<f64>, initial_probs: Option<Vec<f64>>) -> Result<Self, Box<dyn Error>> {

        let mut transition_matrix = Vec::with_capacity(transition_probs.dim().0);
        for probs in transition_probs.axis_iter(Axis(0)) {
            transition_matrix.push(DiscreteDistribution::new(probs.to_vec())?);
        }
	let initial_distribution = match initial_probs {
	    None => DiscreteDistribution::new(calc_steady_state_dist(&transition_probs)?)?,
	    Some(dist) => DiscreteDistribution::new(dist)?,
	};
        Ok(MarkovDNA {
            initial_distribution,
            transition_matrix,
        })
    }

    pub fn generate<R: Rng>(&mut self, length: usize, rng: &mut R) -> Dna {
	let mut dna = Dna {
	    seq: Vec::with_capacity(length),
	};
        if length == 0 {
            return dna;
        }

        let mut current_state = self.initial_distribution.generate(rng);
        dna.seq.push(NUCLEOTIDES[current_state]);

        for _ in 1..length {
            current_state = self.transition_matrix[current_state].generate(rng);
            dna.seq.push(NUCLEOTIDES[current_state]);
        }
        dna
    }
}

pub fn calc_steady_state_dist(transition_matrix: &Array2<f64>) -> Result<Vec<f64>, Box<dyn Error>> {
    let (eig, eigv) = transition_matrix.eig().map_err(|_| "Eigen decomposition failed")?;
    for i in 0..4 {
	if (eig[i].re - 1.).abs() < 1e-6 {
	    let col = eigv.column(i);
	    let sum: f64 = col.mapv(|x| x.re).sum();
	    return Ok(col.mapv(|x| x.re / sum).to_vec());
	}
    }
    Err("No suitable eigenvector found")?
}



pub fn add_errors<R: Rng>(dna: &mut Dna, error_rate: f64, rng: &mut R) {
    let uniform = Uniform::new(0.0, 1.0);
    let random_nucleotide = Uniform::new_inclusive(0, 3);

    for nucleotide in dna.seq.iter_mut() {
        if uniform.sample(rng) < error_rate {
            *nucleotide = NUCLEOTIDES[random_nucleotide.sample(rng)];
        }
    }
}
