// Define DNA and AminoAcid structure + basic operations on the sequences
use anyhow::{anyhow, Result};
use bio::alignment::{pairwise, Alignment};
use phf::phf_map;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use std::fmt;

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
pub const NUCLEOTIDES: [u8; 5] = [b'A', b'C', b'G', b'T', b'N'];
pub static NUCLEOTIDES_INV: phf::Map<u8, usize> = phf_map! {
    b'A' => 0, b'T' => 3, b'G' => 2, b'C' => 1, b'N' => 4,
};

static COMPLEMENT: phf::Map<u8, u8> = phf_map! {
    b'A' => b'T', b'T' => b'A', b'G' => b'C', b'C' => b'G', b'N' => b'N',
};

// pub fn nucleotides_inv(n: u8) -> usize {
//     match n {
//         b'A' => 0,
//         b'T' => 3,
//         b'G' => 2,
//         b'C' => 1,
//         b'N' => 4,
//         _ => panic!("Wrong nucleotide type"),
//     }
// }
// potentially faster
pub fn nucleotides_inv(n: u8) -> usize {
    static LOOKUP_TABLE: [usize; 256] = {
        let mut table = [0; 256];
        table[b'A' as usize] = 0;
        table[b'T' as usize] = 3;
        table[b'G' as usize] = 2;
        table[b'C' as usize] = 1;
        table[b'N' as usize] = 4;
        table
    };

    LOOKUP_TABLE[n as usize]
}
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
pub struct AlignmentParameters {
    // Structure containing all the parameters for the alignment
    // of the V and J genes
    pub min_score_v: i32,
    pub min_score_j: i32,
    pub max_error_d: usize,
}

impl Default for AlignmentParameters {
    fn default() -> AlignmentParameters {
        AlignmentParameters {
            min_score_v: 0,
            min_score_j: 0,
            max_error_d: 100,
        }
    }
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl AlignmentParameters {
    #[new]
    pub fn py_new(min_score_v: i32, min_score_j: i32, max_error_d: usize) -> Self {
        Self::new(min_score_v, min_score_j, max_error_d)
    }
}

impl AlignmentParameters {
    pub fn new(min_score_v: i32, min_score_j: i32, max_error_d: usize) -> Self {
        Self {
            min_score_v,
            min_score_j,
            max_error_d,
        }
    }
}

impl AlignmentParameters {
    fn get_scoring(&self) -> pairwise::Scoring<Box<dyn Fn(u8, u8) -> i32>> {
        pairwise::Scoring {
            gap_open: -50,
            gap_extend: -10,
            // TODO: deal better with possible IUPAC codes
            match_fn: Box::new(|a: u8, b: u8| {
                if a == b {
                    6i32
                } else if (a == b'N') | (b == b'N') {
                    0i32
                } else {
                    -6i32
                }
            }),
            match_scores: None,
            xclip_prefix: 0,
            xclip_suffix: pairwise::MIN_SCORE,
            yclip_prefix: pairwise::MIN_SCORE,
            yclip_suffix: 0,
        }
    }

    pub fn valid_v_alignment(&self, al: &Alignment) -> bool {
        al.score > self.min_score_v && al.xend - al.xstart == al.yend - al.ystart
    }

    pub fn valid_j_alignment(&self, al: &Alignment) -> bool {
        // right now: no insert
        al.score > self.min_score_j && al.xend - al.xstart == al.yend - al.ystart
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Dna {
    pub seq: Vec<u8>,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug, PartialEq)]
pub struct AminoAcid {
    pub seq: Vec<u8>,
}

impl fmt::Display for Dna {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(&self.seq))
    }
}

impl fmt::Display for AminoAcid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(&self.seq))
    }
}

impl Dna {
    pub fn new() -> Dna {
        Dna { seq: Vec::new() }
    }
    pub fn from_string(s: &str) -> Result<Dna> {
        for &byte in s.as_bytes() {
            if !NUCLEOTIDES_INV.contains_key(&byte) {
                // Handle the error if the byte is not in the map
                return Err(anyhow!(format!("Invalid byte: {}", byte)));
            }
        }

        Ok(Dna {
            seq: s.as_bytes().to_vec(),
        })
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl Dna {
    pub fn get_string(&self) -> String {
        self.to_string()
    }

    pub fn translate(&self) -> Result<AminoAcid> {
        if self.seq.len() % 3 != 0 {
            return Err(anyhow!("Translation not possible, invalid length."))?;
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

    pub fn len(&self) -> usize {
        self.seq.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }

    pub fn extend(&mut self, dna: &Dna) {
        self.seq.extend(dna.seq.iter());
    }

    pub fn reverse(&mut self) {
        self.seq.reverse();
    }
}

impl Dna {
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
        // Return dna[start:end]
        Dna {
            seq: self.seq[start..end].to_vec(),
        }
    }

    /// Return dna[start:end] but padded with N if start < 0 or end >= dna.len()
    ///```
    /// use ihor;
    ///let a = ihor::Dna::from_string("ACCAAATGC").unwrap();
    ///assert!(a.extract_padded_subsequence(2, 5).get_string() == "CAA".to_string());
    ///assert!(a.extract_padded_subsequence(-1, 5).get_string() == "NACCAA".to_string());
    ///assert!(a.extract_padded_subsequence(5, 10).get_string() == "ATGCN".to_string());
    ///assert!(a.extract_padded_subsequence(-2, 11).get_string() == "NNACCAAATGCNN".to_string());
    ///```
    pub fn extract_padded_subsequence(&self, start: i64, end: i64) -> Dna {
        let len = self.len() as i64;
        let valid_start = std::cmp::max(0, start) as usize;
        let valid_end = std::cmp::min(len, end) as usize;
        let mut result = Vec::with_capacity((end - start).abs() as usize);

        if start < 0 {
            result.resize(start.unsigned_abs() as usize, b'N');
        }

        if start < len {
            result.extend_from_slice(&self.seq[valid_start..valid_end]);
        }

        if end > len {
            result.resize(result.len() + (end - len) as usize, b'N');
        }

        Dna { seq: result }
    }
    pub fn align_left_right(
        sleft: &Dna,
        sright: &Dna,
        align_params: &AlignmentParameters,
    ) -> Alignment {
        // Align two sequences with this format
        // sleft  : ACATCCCACCATTCA
        // sright :         CCATGACTCATGAC

        let mut aligner = pairwise::Aligner::with_capacity_and_scoring(
            sleft.len(),
            sright.len(),
            align_params.get_scoring(),
        );

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
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl AminoAcid {
    #[staticmethod]
    #[pyo3(name = "from_string")]
    pub fn py_from_string(s: &str) -> AminoAcid {
        AminoAcid::from_string(s)
    }
}

pub fn difference_as_i64(a: usize, b: usize) -> i64 {
    if a >= b {
        // don't check for overflow, trust the system
        // assert!(a - b <= i64::MAX as usize, "Overflow occurred");
        (a - b) as i64
    } else {
        // don't check for underflow either
        // assert!(b - a <= i64::MAX as usize, "Underflow occurred");
        -((b - a) as i64)
    }
}

// pyo3 boiler code
#[cfg(feature = "py_binds")]
#[pymethods]
impl Dna {
    #[new]
    pub fn py_new() -> Dna {
        Dna::new()
    }

    #[staticmethod]
    #[pyo3(name = "from_string")]
    pub fn py_from_string(s: &str) -> Result<Dna> {
        Dna::from_string(s)
    }
}

pub fn count_differences<T: PartialEq>(vec1: &[T], vec2: &[T]) -> usize {
    vec1.iter()
        .zip(vec2.iter())
        .filter(|&(a, b)| a != b)
        .count()
}
