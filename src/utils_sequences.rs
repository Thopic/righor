// Define DNA and AminoAcid structure + basic operations on the sequences
use bio::alignment::{pairwise, Alignment};
use phf::phf_map;

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
pub const NUCLEOTIDES: [u8; 4] = [b'A', b'C', b'G', b'T'];
pub static NUCLEOTIDES_INV: phf::Map<u8, usize> = phf_map! {
    b'A' => 0, b'T' => 3, b'G' => 2, b'C' => 1
};

static COMPLEMENT: phf::Map<u8, u8> = phf_map! {
    b'A' => b'T', b'T' => b'A', b'G' => b'C', b'C' => b'G'
};

#[derive(Default, Clone, Debug)]
pub struct AlignmentParameters {
    // Structure containing all the parameters for the alignment
    // of the V and J genes
    pub min_score_v: i32,
    pub min_score_j: i32,
    pub max_error_v: usize,
    pub max_error_j: usize,
    pub max_error_d: usize,
}

impl AlignmentParameters {
    fn get_scoring(&self) -> pairwise::Scoring<Box<dyn Fn(u8, u8) -> i32>> {
        // these parameters are a bit ad hoc
        // TODO: take into account the fuzzy alignment on the right
        pairwise::Scoring {
            gap_open: -30,
            gap_extend: -2,
            match_fn: Box::new(|a: u8, b: u8| if a == b { 6i32 } else { -6i32 }), // TODO: deal better with possible IUPAC codes
            match_scores: None,
            xclip_prefix: 0,
            xclip_suffix: pairwise::MIN_SCORE,
            yclip_prefix: pairwise::MIN_SCORE,
            yclip_suffix: 0,
        }
    }

    pub fn valid_v_alignment(&self, al: &Alignment) -> bool {
        return al.score > self.min_score_v;
    }

    pub fn valid_j_alignment(&self, al: &Alignment) -> bool {
        return al.score > self.min_score_j;
    }
}

#[derive(Default, Clone, Debug)]
pub struct Dna {
    pub seq: Vec<u8>,
}

#[derive(Default, Clone, Debug)]
pub struct AminoAcid {
    pub seq: Vec<u8>,
}

impl Dna {
    pub fn new() -> Dna {
        Dna { seq: Vec::new() }
    }

    pub fn from_string(s: &str) -> Dna {
        return Dna {
            seq: s.as_bytes().to_vec(),
        };
    }

    pub fn to_string(&self) -> String {
        return String::from_utf8_lossy(&self.seq).to_string();
    }

    pub fn translate(&self) -> Result<AminoAcid, &'static str> {
        if self.seq.len() % 3 != 0 {
            return Err("Translation not possible, invalid length.");
        }

        let amino_sequence: Vec<u8> = self
            .seq
            .chunks(3)
            .filter_map(|codon| {
                let codon_str = std::str::from_utf8(codon).ok()?;
                DNA_TO_AMINO.get(codon_str).copied()
            })
            .collect();
        Ok(AminoAcid {
            seq: amino_sequence,
        })
    }
    pub fn reverse_complement(&self) -> Dna {
        Dna {
            seq: self
                .seq
                .iter()
                .filter_map(|x| COMPLEMENT.get(x).copied())
                .rev()
                .collect(),
        }
    }

    pub fn extract_subsequence(&self, start: usize, end: usize) -> Dna {
        Dna {
            seq: self.seq[start..end].to_vec(),
        }
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

    pub fn align_left_right(
        sleft: &Dna,
        sright: &Dna,
        align_params: &AlignmentParameters,
    ) -> Alignment {
        // Align two sequences with this format
        // ACATCCCACCATTCA
        //         CCATGACTCATGAC

        let scoring = align_params.get_scoring();
        let mut aligner =
            pairwise::Aligner::with_capacity_and_scoring(sleft.len(), sright.len(), scoring);

        aligner.custom(sleft.seq.as_slice(), sright.seq.as_slice())
    }

    pub fn position_differences(sequence: &Dna, template: &Dna) -> Vec<usize> {
        // Return the position of the differences between sequence
        // and template
        template
            .seq
            .iter()
            .zip(sequence.seq.iter())
            .enumerate()
            .filter_map(
                |(index, (&v1, &v2))| {
                    if v1 != v2 {
                        Some(index)
                    } else {
                        None
                    }
                },
            )
            .collect()
    }
}

impl AminoAcid {
    pub fn from_string(s: &str) -> AminoAcid {
        return AminoAcid {
            seq: s.as_bytes().to_vec(),
        };
    }
    pub fn to_string(&self) -> String {
        return String::from_utf8_lossy(&self.seq).to_string();
    }
}

// pub fn differences<T: PartialEq>(vec1: &[T], vec2: &[T]) -> Vec<usize> {
//     vec1.iter()
//         .zip(vec2.iter())
//         .enumerate()
//         .filter_map(|(index, (a, b))| if a != b { Some(index) } else { None })
//         .collect()
// }

pub fn differences<T, I1, I2>(iter1: I1, iter2: I2) -> Vec<usize>
where
    T: PartialEq,
    I1: IntoIterator<Item = T>,
    I2: IntoIterator<Item = T>,
{
    iter1
        .into_iter()
        .zip(iter2.into_iter())
        .enumerate()
        .filter_map(|(index, (a, b))| if a != b { Some(index) } else { None })
        .collect()
}
