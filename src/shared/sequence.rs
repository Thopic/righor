/// Contains the basic struct and function for loading and aligning sequences
use crate::shared::utils::count_differences;
use crate::shared::AlignmentParameters;
use crate::vdj::model::Model as ModelVDJ;
use anyhow::{anyhow, Result};
use bio::alignment::{pairwise, Alignment};
use phf::phf_map;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

// pub trait DnaLike {
//     pub fn len(&self) -> usize;
//     pub fn is_empty(&self) -> bool;
//     pub fn reverse_complement(&self) -> Self where Self: Sized;
//     pub fn extract_subsequence(&self, start: usize, end: usize) -> Self where Self: Sized;
//     pub fn extract_padded_subsequence(&self, start: i64, end: i64) -> Self where Self: Sized;
//     pub fn probability(&self, markov_coefficients: Array2<f64>) -> f64;
// }

// /// Partially defined dna sequence created from an amino-acid sequence
// struct UndefinedDna {
//     // List of codons
//     codons: Vec<u8>,
//     // the start of the actual nucleotide sequence (potentially mid-codon)
//     // belong to  0..=2
//     codon_start: usize,
//     // the end of the actual nucleotide sequence (potentially mid-codon)
//     // belong to 0..2
//     codon_end: usize,
//     // true if the sequence considered is the reverse complement of the codons stored
//     reverse_complement: bool,

// }

// impl UndefinedDna {
//     pub fn pad_right(&mut self, n: usize) {
// 	// X for undefined amino-acid
// 	let number_aa_added = (n - self.codon_end)/3;
// 	self.codons.resize(number_aa_added, b'X');
// 	self.codon_end = (n - self.codon_end)%3;
//     }

//     pub fn pad_left(&mut self, n: usize) {
// 	// NNN corresponds to all possible amino-acid, so to the bitvec (2^64 - 1)
// 	let number_aa_added = (n - self.codon_start)/3;
// 	self.codons.splice(0..0, vec![b'X', number_aa_added]);
// 	self.codon_start = (n - self.codon_start)%3;
//     }

//     /// Make an amino-acid sequence into an "UndefinedDna" sequence
//     pub fn from_aminoacid(aa: &AminoAcid) -> UndefinedDna {
// 	UndefinedDna {
// 	    codons: aa.clone(),
// 	    codon_start: 0,
// 	    codon_end: 0,
// 	}
//     }

//     /// lossy process, remove some information about the codon
//     pub fn to_dna(&self) -> Dna {

// 	let sequence = Dna {
// 	    seq: self.codons.iter().flat_map(|&aa| AMINO_TO_DNA_LOSSY.get(&aa).unwrap().clone()).collect(),
// 	};
// 	sequence.extract_subsequence(self.codon_start, self.len() - self.codon_end)
//     }

// }

// impl DnaLike for UndefinedDna {

//     /// Return the reverse complement of the sequence
//     pub fn reverse_complement(&mut self) -> UndefinedDna {

//     }

//     /// Extract a subsequence from the dna. No particular checks.
//     pub fn extract_subsequence(&self, start: usize, end: usize) -> UndefinedDna {
// 	// where to start in the amino-acid sequence

// 	// example:
// 	// start = 10, end = 20
// 	// codon_start = 2, codon_end = 1
// 	//   <---------------------------------->  : true sequence
// 	// ....................................... : full stored data
// 	//  x  x  x  x  x  x  x  x  x  x  x  x  x  : amino-acids
// 	//             <-------->                  : extracted sequence
// 	//             ............                : stored data for extracted sequence
// 	//

// 	let shift_start = start + self.codon_start;
// 	let shift_end = end + self.codon_start;

// 	let aa_start = shift_start / 3;
// 	let aa_end = shift_end / 3 + 1;

// 	UndefinedDna {
// 	    codons: self.codons[aa_start..aa_end].to_vec(),
// 	    codon_start: shift_start % 3,
// 	    codon_end: 3*(aa_end) - shift_end

// 	}
//     }

//     /// Return dna[start:end] but padded with N if start < 0 or end >= dna.len()
//     pub fn extract_padded_subsequence(&self, start: i64, end: i64) -> UndefinedDna {

// 	// example:
// 	// start = -4, end = 17
// 	// codon_start = 2, codon_end = 0
// 	//               0    '    '   '
// 	//               <----------->             : true sequence
// 	//             ...............             : full stored data
// 	//              x  x  x  x  x              : amino-acids
// 	//           <-------------------->        : extracted and padded sequence
// 	//          ........................       : stored data for padded sequence

// 	let result = self.clone();
//         let len = self.len() as i64;
// 	let mut shift = 0;

// 	if start < 0 {
// 	    result.pad_left(start.unsigned_abs());
// 	    shift = start.unsigned_abs();
// 	}
// 	if end > self.len() {
// 	    result.pad_right(end - self.len());
// 	}
// 	result.extract_subsequence(start + shift, end + shift)
//     }

//     pub fn len(&self) -> usize {
// 	3*self.codons.len() - self.codon_start - self.codon_end
//     }

//     pub fn is_empty(&self) -> bool {
// 	self.len() == 0
//     }
// }

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
pub const NUCLEOTIDES: [u8; 15] = [
    b'A', b'C', b'G', b'T', b'N', b'R', b'Y', b'S', b'W', b'K', b'M', b'B', b'D', b'H', b'V',
];
pub static NUCLEOTIDES_INV: phf::Map<u8, usize> = phf_map! {
    b'A' => 0, b'T' => 3, b'G' => 2, b'C' => 1, b'N' => 4,
    b'R' => 5, b'Y' => 6, b'S' => 7, b'W' => 8, b'K' => 9,
    b'M' => 10, b'B' => 11, b'D' => 12, b'H' => 13, b'V' => 14,
};

static COMPLEMENT: phf::Map<u8, u8> = phf_map! {
    b'A' => b'T', b'T' => b'A', b'G' => b'C', b'C' => b'G', b'N' => b'N',
    b'R' => b'Y', b'Y' => b'R', b'S' => b'S', b'W' => b'W', b'K' => b'M',
    b'M' => b'K', b'B' => b'A', b'D' => b'C', b'H' => b'G', b'V' => b'T',

};

// static AMINO_TO_DNA_LOSSY: phf::Map<&'static str, u8> = phf_map! {
//     b'A' => "GCN",
//     b'C' => "TGY",
//     b'D' => "GAY",
//     b'E' => "GAR",
//     b'F' => "TTY",
//     b'G' => "GGN",
//     b'H' => "CAY",
//     b'I' => "ATH",
//     b'L' => "YTN",
//     b'K' => "AAR",
//     b'M' => "ATG",
//     b'N' => "AAY",
//     b'P' => "CCN",
//     b'Q' => "CAR",
//     b'R' => "MGN",
//     b'S' => "WSN",
//     b'T' => "CAN",
//     b'V' => "GTN",
//     b'W' => "TGG",
//     b'Y' => "TAY",
//     b'*' => "TRR",
// };

pub fn nucleotides_inv(n: u8) -> usize {
    static LOOKUP_TABLE: [usize; 256] = {
        let mut table = [0; 256];
        table[b'A' as usize] = 0;
        table[b'T' as usize] = 3;
        table[b'G' as usize] = 2;
        table[b'C' as usize] = 1;
        table[b'N' as usize] = 4;
        table[b'R' as usize] = 5;
        table[b'Y' as usize] = 6;
        table[b'S' as usize] = 7;
        table[b'W' as usize] = 8;
        table[b'K' as usize] = 9;
        table[b'M' as usize] = 10;
        table[b'B' as usize] = 11;
        table[b'D' as usize] = 12;
        table[b'H' as usize] = 13;
        table[b'V' as usize] = 14;
        table
    };

    LOOKUP_TABLE[n as usize]
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq, Eq)]
pub struct Dna {
    pub seq: Vec<u8>,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug, PartialEq, Eq)]
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
    fn __repr__(&self) -> String {
        self.get_string()
    }

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
    /// use righor;
    ///let a = righor::Dna::from_string("ACCAAATGC").unwrap();
    ///assert!(a.extract_padded_subsequence(2, 5).get_string() == "CAA".to_string());
    ///assert!(a.extract_padded_subsequence(-1, 5).get_string() == "NACCAA".to_string());
    ///assert!(a.extract_padded_subsequence(5, 10).get_string() == "ATGCN".to_string());
    ///assert!(a.extract_padded_subsequence(-2, 11).get_string() == "NNACCAAATGCNN".to_string());
    ///```
    pub fn extract_padded_subsequence(&self, start: i64, end: i64) -> Dna {
        let len = self.len() as i64;
        let valid_start = std::cmp::max(0, start) as usize;
        let valid_end = std::cmp::min(len, end) as usize;
        let mut result = Vec::with_capacity((end - start).unsigned_abs() as usize);

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

    // A fast alignment algorithm just for V (because V is a bit long)
    // Basically ignore the possible insertions/deletions
    // seq     :    SSSSSSSSSSSSSSSSSSSSSSSSSSS
    // cutV    :              VVVV (start_vcut = 13, leftv_cutoff = 4)
    // full V  : VVVVVVVVVVVVVVVVV
    //              ^   ystart = 0, xstart = 3
    //                           ^ yend = 13, xend = 16
    //                        ^ cutal.xstart = 0, cutal.ystart = 10
    //                           ^ cutal.xend=3, cutal.yend = 13
    pub fn v_alignment(
        v: &Dna,
        seq: &Dna,
        align_params: &AlignmentParameters,
    ) -> Option<Alignment> {
        let start_vcut = if v.len() > align_params.left_v_cutoff {
            v.len() - align_params.left_v_cutoff
        } else {
            0
        };

        if start_vcut == 0 {
            // just do a normal alignment
            let alignment = Self::align_left_right(v, seq, align_params);
            if !align_params.valid_v_alignment(&alignment) {
                return None;
            }
            return Some(alignment);
        }

        // Align just the end of the V gene (faster)
        let cutv = &v.seq[start_vcut..];

        let mut aligner = pairwise::Aligner::with_capacity_and_scoring(
            cutv.len(),
            seq.len(),
            align_params.get_scoring_local(), // no left-right constraint this time
        );

        let cutal = aligner.custom(cutv, seq.seq.as_slice());
        // V should start before the sequence
        if cutal.ystart > start_vcut {
            return None;
        }

        let alignment = bio::alignment::Alignment {
            ystart: 0, // that's where V start in the sequence, so always 0
            xstart: start_vcut + cutal.xstart - cutal.ystart,
            // this doesn't work with indels
            xend: start_vcut + cutal.xend,
            yend: cutal.yend,
            ylen: seq.len(),
            xlen: v.len(),
            ..Default::default() // the other values are meaningless in that context
        };

        if !align_params.valid_v_alignment(&alignment) {
            return None;
        }

        Some(alignment)
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
    fn __repr__(&self) -> String {
        String::from_utf8_lossy(&self.seq).to_string()
    }

    #[staticmethod]
    #[pyo3(name = "from_string")]
    pub fn py_from_string(s: &str) -> AminoAcid {
        AminoAcid::from_string(s)
    }
}

// pyo3 boiler code
#[cfg(feature = "py_binds")]
#[pymethods]
impl Dna {
    #[new]
    #[pyo3(signature = (sequence = ""))]
    pub fn py_new(sequence: &str) -> Result<Dna> {
        Dna::from_string(sequence)
    }

    #[staticmethod]
    #[pyo3(name = "from_string")]
    pub fn py_from_string(s: &str) -> Result<Dna> {
        Dna::from_string(s)
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug)]
pub struct VJAlignment {
    // Structure containing the alignment between a V/J gene and the sequence
    // Note that the genes defined here include the palindromic insertion at the end

    // Example
    // gene (V):  ATACGATCATTGACAATCTGGAGATACGTA
    //                         ||||||\|||||\|\\\
    // sequence:               CAATCTAGAGATTCCAATCTAGAGATTCA
    // start_gene -------------^ 13
    // end_gene   ------------------------------^ 30
    // start_seq               0
    // end_seq                 -----------------^ 17
    // errors[u] contains the number of errors left if u nucleotides are removed
    // (starting from the end of the V-gene, or the beginning of the J gene)
    // the length of this vector is the maximum number of v/j deletion.
    // so here [5,4,3,2,2,1,1,1,1,1,1,0,0,...]
    // score is the score of the alignment according to the alignment process
    pub index: usize, // index of the gene in the model
    pub start_seq: usize,
    pub end_seq: usize,
    pub start_gene: usize,
    pub end_gene: usize,
    pub errors: Vec<usize>,
    pub score: i32,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl VJAlignment {
    pub fn nb_errors(&self, del: usize) -> usize {
        if del >= self.errors.len() {
            return match self.errors.last() {
                None => 0,
                Some(l) => *l,
            };
        }
        self.errors[del]
    }

    pub fn length_with_deletion(&self, del: usize) -> usize {
        // just return the aligned part length (that matches the seq)
        if self.end_seq <= self.start_seq + del {
            return 0;
        }
        self.end_seq - self.start_seq - del
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
#[derive(Clone, Debug)]
pub struct DAlignment {
    // Structure containing the alignment between a D gene and the sequence
    // Similarly to VJaligment the gene include the palindromic insertion
    // d-gene  :         DDDDDDDDDDDD
    // sequence:     SSSSSSSSSSSSSSSSSSSSS
    // pos     ----------^ 4
    // error_left contains the number of errors in the alignment starting from the left
    // error_right contains the number of errors in the alignment starting from the right
    // For example:
    // DDDDDDDDDDD
    // SXSSSXSSXSX
    // errors_left:  [4,4,3,3,3,3,2,2,2,1,1]
    // errors_right: [4,3,3,2,2,2,1,1,1,1,0]
    // "pos" represents the start of the sequence *with palindromes added*
    pub index: usize,
    pub len_d: usize, // length of the D gene (with palindromic inserts)
    pub pos: usize,   // begining of the D-gene in the sequence (can't be < 0)

    pub dseq: Arc<Dna>,     // the sequence of the D gene
    pub sequence: Arc<Dna>, // the complete sequence aligned
}

//#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl DAlignment {
    pub fn nb_errors(&self, deld5: usize, deld3: usize) -> usize {
        if deld5 + deld3 > self.len_d {
            return 0;
        }

        count_differences(
            &self.sequence.seq[self.pos + deld5..self.pos + self.dseq.len() - deld3],
            &self.dseq.seq[deld5..self.dseq.len() - deld3],
        )
    }
    pub fn length_with_deletion(&self, deld5: usize, deld3: usize) -> usize {
        self.len() - deld5 - deld3
    }

    pub fn len(&self) -> usize {
        self.len_d
    }
    pub fn is_empty(&self) -> bool {
        self.len_d == 0
    }

    pub fn display(&self, sequence: &Dna, model: &ModelVDJ) -> String {
        let mut line1 = "".to_string();
        let mut line2 = "".to_string();
        let mut line3 = "".to_string();
        let dna_sequence = sequence.seq.clone();
        let dna_dgene = model.seg_ds[self.index].seq_with_pal.clone().unwrap().seq;
        for ii in 0..sequence.len() {
            line1 += &(dna_sequence[ii] as char).to_string();
            if (ii < self.pos) || (ii >= self.pos + self.len_d) {
                line2 += " ";
                line3 += " ";
            } else {
                line3 += &(dna_dgene[ii - self.pos] as char).to_string();
                if dna_dgene[ii - self.pos] != dna_sequence[ii] {
                    line2 += "\\";
                } else {
                    line2 += "|";
                }
            }
        }
        format!("{line1}\n{line2}\n{line3}\n")
    }
}
