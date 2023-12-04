use phf::phf_map;
use rand::Rng;
use rand::distributions::Distribution;
use rand_distr::WeightedAliasIndex;

// maximal distance between sequences, used as a default return value
// when the distances are above the threshold.
const MAX_DISTANE: usize = 10000;



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




#[derive(Default, Clone, Debug)]
pub struct Dna{
    pub seq: Vec<u8>,
}

#[derive(Default, Clone, Debug)]
pub struct AminoAcid{
    pub seq: Vec<u8>,
}


impl Dna{

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
}


// Define the random distributions used in the rest of the code.


// Generate an integer with a given probability
struct DiscreteDistribution {
    distribution: WeightedAliasIndex<f64>,
}


impl DiscreteDistribution {
    pub fn new(weights: Vec<f64>) -> Result<Self, rand_distr::WeightedError> {
        let distribution = WeightedAliasIndex::new(weights)?;
        Ok(DiscreteDistribution { distribution })
    }

    pub fn generate<R: Rng>(&mut self, rng: &mut R) -> usize {
        self.distribution.sample(rng)
    }
}


// Narkov chain structure (for the insertion process)
struct MarkovDNA {
    initial_distribution: DiscreteDistribution, // first nucleotide, ACGT order
    transition_matrix: Vec<DiscreteDistribution>, // Markov matrix, ACGT order
}


impl MarkovDNA {
    pub fn new(initial_probs: Vec<f64>, transition_probs: Vec<Vec<f64>>) -> Result<Self, rand_distr::WeightedError> {
        let initial_distribution = DiscreteDistribution::new(initial_probs)?;
        let mut transition_matrix = Vec::with_capacity(transition_probs.len());
        for probs in transition_probs {
            transition_matrix.push(DiscreteDistribution::new(probs)?);
        }

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
