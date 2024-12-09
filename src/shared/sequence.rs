//! Contains the basic struct and function for loading and aligning sequences
use crate::shared::amino_acids::{DegenerateCodon, DegenerateCodonSequence};

use crate::shared::AlignmentParameters;
use anyhow::{anyhow, Result};
use bio::alignment::{pairwise, Alignment};
use phf::phf_map;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};

/////////////////////////////////////
// Constants

// Unknown nucleotide
// Not the same thing as N, N means everything is possible,
// while BLANK means "there was only one but unknown".
// important when computing probabilities
// For example, if we want to compute the likelihood of the CDR3
// CAGACNCTA, then we want to have 'N', because the expected end result
// is the sum over the proba of all CAGAC*CTA.
// But if in the inference we miss a bit of the sequence and end up with
// an insertion like "...CA", we don't want to compute the likelihood of
// this insertion by summing over all probabilities of AAACA up to TTTCA
// at best this insertion has a probability max(***CA).
// I'm not dealing well with this now. TODO.
pub const BLANK: usize = 4;
pub const BLANKN: u8 = b'N';

// The standard ACGT nucleotides
// R: A/G, Y: T/C, S: C/G, W:A/T, M: C/A, K: G/T, B:T/C/G, D:A/G/T, H:A/C/T, V:A/C/G,
pub const NUCLEOTIDES: [u8; 15] = [
    b'A', b'C', b'G', b'T', b'N', b'R', b'Y', b'S', b'W', b'K', b'M', b'B', b'D', b'H', b'V',
];

pub const AMINOACIDS: [u8; 21] = [
    b'A', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'L', b'K', b'M', b'N', b'P', b'Q', b'R', b'S',
    b'T', b'V', b'W', b'Y', b'*',
];

pub const ALL_POSSIBLE_CODONS_AA: [u8; 85] = [
    b'A', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'L', b'K', b'M', b'N', b'P', b'Q', b'R', b'S',
    b'T', b'V', b'W', b'Y', b'*', 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
    179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
];

static AMINO_TO_DNA_LOSSY: phf::Map<u8, [u8; 3]> = phf_map! {
    b'A' => [b'G', b'C', b'N'],
    b'C' => [b'T', b'G', b'Y'],
    b'D' => [b'G', b'A', b'Y'],
    b'E' => [b'G', b'A', b'R'],
    b'F' => [b'T', b'T', b'Y'],
    b'G' => [b'G', b'G', b'N'],
    b'H' => [b'C', b'A', b'Y'],
    b'I' => [b'A', b'T', b'H'],
    b'L' => [b'Y', b'T', b'N'],
    b'K' => [b'A', b'A', b'R'],
    b'M' => [b'A', b'T', b'G'],
    b'N' => [b'A', b'A', b'Y'],
    b'P' => [b'C', b'C', b'N'],
    b'Q' => [b'C', b'A', b'R'],
    b'R' => [b'M', b'G', b'N'],
    b'S' => [b'W', b'S', b'N'],
    b'T' => [b'A', b'C', b'N'],
    b'V' => [b'G', b'T', b'N'],
    b'W' => [b'T', b'G', b'G'],
    b'Y' => [b'T', b'A', b'Y'],
    b'*' => [b'T', b'R', b'R'],
};

// Add the situations where the codon is fully known
fn amino_to_dna_lossy(x: u8) -> [u8; 3] {
    if x < b'Z' {
        AMINO_TO_DNA_LOSSY[&x]
    } else {
        let c1 = ((x - 128) % 4) as usize;
        let c2 = (((x - 128) / 4) % 4) as usize;
        let c3 = ((x - 128) / 16) as usize;
        [NUCLEOTIDES[c1], NUCLEOTIDES[c2], NUCLEOTIDES[c3]]
    }
}

/////////////////////////////////////

/////////////////////////////////////
// Structures
/////////////////////////////////////

#[cfg_attr(
    all(feature = "py_binds", feature = "pyo3"),
    pyclass(get_all, set_all, eq, eq_int)
)]
#[derive(Clone, Debug, Copy, Default, PartialEq)]
pub enum SequenceType {
    #[default]
    Dna,
    Protein,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
/// Needed because pyo3 cannot deal with enum directly
pub struct DnaLike {
    inner: DnaLikeEnum,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
/// Enum that deals with the different type of Dna
/// will use different algorithms depending on the type of Dna
pub enum DnaLikeEnum {
    Known(Dna),         // known Dna (A/T/G/C)
    Ambiguous(Dna),     // degenerate Dna (contains degenerate nucleotides N/H...)
    Protein(AminoAcid), // reverse-translated amino-acid sequence
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AminoAcid {
    pub seq: Vec<u8>,
    pub start: usize, // the start of the true sequence within the 1st codon (>0 <3)
    pub end: usize,   // the start of the true sequence within the last codon (>0 <3)
}

/// Dna sequence (for A/T/G/C, but also used internally for degenerate nucleotides)
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
pub struct Dna {
    pub seq: Vec<u8>,
}

/////////////////////////////////////
// Implementations
/////////////////////////////////////

impl Hash for Dna {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_string().hash(state);
    }
}

impl From<DnaLike> for DnaLikeEnum {
    fn from(dnalike: DnaLike) -> Self {
        dnalike.inner
    }
}

impl Default for DnaLikeEnum {
    fn default() -> Self {
        DnaLikeEnum::Known(Dna::default())
    }
}

impl From<Dna> for DnaLikeEnum {
    fn from(dna: Dna) -> Self {
        DnaLikeEnum::from_dna(dna)
    }
}

impl From<AminoAcid> for DnaLikeEnum {
    fn from(aminoacid: AminoAcid) -> Self {
        DnaLikeEnum::from_amino_acid(aminoacid)
    }
}

impl From<DnaLikeEnum> for DnaLike {
    fn from(dnalike: DnaLikeEnum) -> Self {
        DnaLike { inner: dnalike }
    }
}

impl From<Dna> for DnaLike {
    fn from(dna: Dna) -> Self {
        DnaLike {
            inner: DnaLikeEnum::from_dna(dna),
        }
    }
}

impl From<AminoAcid> for DnaLike {
    fn from(aminoacid: AminoAcid) -> Self {
        DnaLike {
            inner: DnaLikeEnum::from_amino_acid(aminoacid),
        }
    }
}

impl DnaLikeEnum {
    pub fn len(&self) -> usize {
        match self {
            DnaLikeEnum::Known(s) | DnaLikeEnum::Ambiguous(s) => s.len(),
            DnaLikeEnum::Protein(s) => 3 * s.seq.len() - s.start - s.end,
        }
    }

    pub fn is_protein(&self) -> bool {
        match self {
            DnaLikeEnum::Known(_s) | DnaLikeEnum::Ambiguous(_s) => false,
            DnaLikeEnum::Protein(_s) => true,
        }
    }

    /// for a sequence of length 2, return all the possible matrix idx
    /// i.e 4*nucleotide[0] + nucleotide[1]
    pub fn to_matrix_idx(&self) -> Vec<usize> {
        match self {
            DnaLikeEnum::Known(s) | DnaLikeEnum::Ambiguous(s) => s.to_matrix_idx(),
            DnaLikeEnum::Protein(s) => s.to_degen_cod_seq().to_matrix_idx(),
        }
    }

    /// Reverse the sequence ATTG -> GTTA.
    pub fn reverse(&mut self) {
        match self {
            DnaLikeEnum::Known(s) | DnaLikeEnum::Ambiguous(s) => s.reverse(),
            DnaLikeEnum::Protein(s) => s.reverse(),
        }
    }

    /// Send the `DnaLikeEnum` to a Dna object. Can be lossy
    pub fn to_dna(&self) -> Dna {
        match self {
            DnaLikeEnum::Known(s) | DnaLikeEnum::Ambiguous(s) => s.clone(),
            DnaLikeEnum::Protein(s) => s.to_dna(),
        }
    }

    pub fn get_string(&self) -> String {
        match self {
            DnaLikeEnum::Known(s) | DnaLikeEnum::Ambiguous(s) => s.get_string(),
            DnaLikeEnum::Protein(s) => s.to_dna().get_string(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            DnaLikeEnum::Known(s) | DnaLikeEnum::Ambiguous(s) => s.is_empty(),
            DnaLikeEnum::Protein(s) => s.is_empty(),
        }
    }

    pub fn from_dna(seq: Dna) -> DnaLikeEnum {
        if seq
            .seq
            .iter()
            .all(|&x| x == b'A' || x == b'T' || x == b'G' || x == b'C')
        {
            DnaLikeEnum::Known(seq)
        } else {
            DnaLikeEnum::Ambiguous(seq)
        }
    }

    pub fn from_amino_acid(seq: AminoAcid) -> DnaLikeEnum {
        DnaLikeEnum::Protein(seq)
    }

    // pub fn extended(&self, other: DnaLikeEnum) -> DnaLikeEnum {
    //     let mut s = self.clone();
    //     s.extend(&other);
    //     s
    // }

    pub fn to_dnas(self) -> Vec<Dna> {
        match self {
            Self::Known(s) => vec![s],
            Self::Ambiguous(s) => s.to_dnas(),
            Self::Protein(s) => s.to_dnas(),
        }
    }

    // TODO: this can be coded better (but probably doesn't matter)
    /// Concatenate self + other and return a `DegenerateCodonsequence` object. Needed because `AminoAcid` + `Dna` may not be a valid AA
    pub fn extended(&self, other: &DnaLikeEnum) -> DegenerateCodonSequence {
        match (self.clone(), other) {
            // Known cases
            (DnaLikeEnum::Known(mut self_dna), DnaLikeEnum::Known(other_dna)) => {
                self_dna.extend(other_dna);
                DegenerateCodonSequence::from_dna(&self_dna.clone(), 0)
            }
            (DnaLikeEnum::Known(self_dna), DnaLikeEnum::Ambiguous(other_dna)) => {
                let mut new_dna = self_dna.clone();
                new_dna.extend(other_dna);
                DegenerateCodonSequence::from_dna(&new_dna, 0)
            }
            (DnaLikeEnum::Known(self_dna), DnaLikeEnum::Protein(other_protein)) => {
                let mut new_protein =
                    DegenerateCodonSequence::from_aminoacid(&other_protein.clone());
                new_protein.append_to_dna(&self_dna);
                new_protein
            }
            // Ambiguous cases
            (DnaLikeEnum::Ambiguous(mut self_dna), DnaLikeEnum::Known(other_dna)) => {
                self_dna.extend(other_dna);
                DegenerateCodonSequence::from_dna(&self_dna.clone(), 0)
            }
            (DnaLikeEnum::Ambiguous(mut self_dna), DnaLikeEnum::Ambiguous(other_dna)) => {
                self_dna.extend(other_dna);
                DegenerateCodonSequence::from_dna(&self_dna.clone(), 0)
            }
            (DnaLikeEnum::Ambiguous(self_dna), DnaLikeEnum::Protein(other_protein)) => {
                let mut dcs = DegenerateCodonSequence::from_aminoacid(&other_protein.clone());
                dcs.append_to_dna(&self_dna);
                dcs
            }
            // Protein cases
            (DnaLikeEnum::Protein(self_protein), DnaLikeEnum::Known(other_dna)) => {
                let mut dcs = DegenerateCodonSequence::from_aminoacid(&self_protein).clone();
                dcs.extend_dna(other_dna);
                dcs
            }
            (DnaLikeEnum::Protein(self_protein), DnaLikeEnum::Ambiguous(other_dna)) => {
                let mut dcs = DegenerateCodonSequence::from_aminoacid(&self_protein);
                dcs.extend_dna(other_dna);
                dcs
            }
            (DnaLikeEnum::Protein(_), DnaLikeEnum::Protein(_)) => {
                panic!("Generally invalid");
            }
        }
    }

    pub fn extended_in_frame(&self, other: &DnaLikeEnum) -> DnaLikeEnum {
        match (self, other) {
            (Self::Known(x), Self::Known(y)) => Self::Known(x.extended(y)),
            (Self::Known(x) | Self::Ambiguous(x), Self::Known(y) | Self::Ambiguous(y)) => {
                Self::Ambiguous(x.extended(y))
            }
            (Self::Known(x), Self::Protein(y)) => Self::Protein(y.append_to_dna_in_frame(x)),
            (Self::Protein(x), Self::Known(y)) => Self::Protein(x.extend_with_dna_in_frame(y)),
            (Self::Protein(x), Self::Protein(y)) => Self::Protein(x.extended(y)),
            (Self::Protein(_x), Self::Ambiguous(_y)) => panic!("Not a valid extension"),
            (Self::Ambiguous(_x), Self::Protein(_y)) => panic!("Not a valid extension"),
        }
    }

    pub fn extract_subsequence(&self, start: usize, end: usize) -> DnaLikeEnum {
        match self {
            DnaLikeEnum::Known(s) => DnaLikeEnum::Known(s.extract_subsequence(start, end)),
            DnaLikeEnum::Ambiguous(s) => DnaLikeEnum::Ambiguous(s.extract_subsequence(start, end)),
            DnaLikeEnum::Protein(s) => DnaLikeEnum::Protein(s.extract_subsequence(start, end)),
        }
    }

    /// Return dna[start:end] but padded with N if `start < 0` or `end >= dna.len()`
    pub fn extract_padded_subsequence(&self, start: i64, end: i64) -> DnaLikeEnum {
        match self {
            DnaLikeEnum::Known(s) => {
                let result = s.extract_padded_subsequence(start, end);
                if result.seq.contains(&b'N') {
                    DnaLikeEnum::Ambiguous(result)
                } else {
                    DnaLikeEnum::Known(result)
                }
            }
            DnaLikeEnum::Ambiguous(s) => {
                let result = s.extract_padded_subsequence(start, end);
                if result
                    .seq
                    .iter()
                    .all(|&x| x == b'A' || x == b'T' || x == b'G' || x == b'C')
                {
                    DnaLikeEnum::Known(result)
                } else {
                    DnaLikeEnum::Ambiguous(result)
                }
            }

            DnaLikeEnum::Protein(s) => {
                DnaLikeEnum::Protein(s.extract_padded_subsequence(start, end))
            }
        }
    }

    pub fn align_left_right(
        sleft: &DnaLikeEnum,
        sright: &DnaLikeEnum,
        align_params: &AlignmentParameters,
    ) -> Alignment {
        Dna::align_left_right(&sleft.to_dna(), &sright.to_dna(), align_params)
    }

    // A fast alignment algorithm just for V (because V is a bit long)
    pub fn v_alignment(
        vgene: &Dna,
        sequence: &DnaLikeEnum,
        align_params: &AlignmentParameters,
    ) -> Option<Alignment> {
        match sequence {
            DnaLikeEnum::Known(seq) | DnaLikeEnum::Ambiguous(seq) => {
                Dna::v_alignment(vgene, seq, align_params)
            }
            DnaLikeEnum::Protein(seq) => Dna::v_alignment(vgene, &seq.to_dna(), align_params),
        }
    }

    /// Count the number of differences between the sequence and the template
    /// Assuming they both start at the same point
    pub fn count_differences(&self, template: &Dna) -> usize {
        match self {
            DnaLikeEnum::Known(s) | DnaLikeEnum::Ambiguous(s) => s.count_differences(template),
            DnaLikeEnum::Protein(s) => s.count_differences(template),
        }
    }

    /// Translate the sequence
    pub fn translate(&self) -> Result<AminoAcid> {
        match self {
            DnaLikeEnum::Known(s) | DnaLikeEnum::Ambiguous(s) => s.translate(),
            DnaLikeEnum::Protein(s) => s.translate(),
        }
    }

    // /// return all possible extremities for a given sequence
    // /// the extremities are encoded in the index form for the 16x16 likelihood matrix
    // /// hence (4*x[0] +  x[1], 4*x[x.len()] + x[x.len()+1])
    // pub fn valid_extremities(&self) -> Vec<(usize, usize)> {
    //     match self {
    //         DnaLikeEnum::Known(s) => s.valid_extremities(),
    //         DnaLikeEnum::Ambiguous(s) => s.valid_extremities(),
    //         DnaLikeEnum::Protein(s) => s.valid_extremities(),
    //     }
    // }
}

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

/// Find the degenerate nucleotide that can match a list of nucleotides
pub fn degenerate_nucleotide(x: &[u8]) -> u8 {
    static MASK_TABLE: [u8; 256] = {
        let mut table = [0; 256];
        table[b'A' as usize] = 0b00000001;
        table[b'C' as usize] = 0b00000010;
        table[b'G' as usize] = 0b00000100;
        table[b'T' as usize] = 0b00001000;
        table[b'N' as usize] = 0b00001111;
        table[b'R' as usize] = 0b00000101; // A or G
        table[b'Y' as usize] = 0b00001010; // T or C
        table[b'S' as usize] = 0b00000110; // G or C
        table[b'W' as usize] = 0b00001001; // A or T
        table[b'K' as usize] = 0b00001100; // G or T
        table[b'M' as usize] = 0b00000011; // A or C
        table[b'B' as usize] = 0b00001110; // C/G/T
        table[b'D' as usize] = 0b00001101; // A/G/T
        table[b'H' as usize] = 0b00001011; // A/C/T
        table[b'V' as usize] = 0b00000111; // A/C/G
        table
    };

    static REVERSE_TABLE: [u8; 256] = {
        let mut table = [0; 256];
        table[0b00000001] = b'A';
        table[0b00000010] = b'C';
        table[0b00000100] = b'G';
        table[0b00001000] = b'T';
        table[0b00001111] = b'N';
        table[0b00000101] = b'R';
        table[0b00001010] = b'Y';
        table[0b00000110] = b'S';
        table[0b00001001] = b'W';
        table[0b00001100] = b'K';
        table[0b00000011] = b'M';
        table[0b00001110] = b'B';
        table[0b00001101] = b'D';
        table[0b00001011] = b'H';
        table[0b00000111] = b'V';
        table
    };

    REVERSE_TABLE[x.iter().fold(0, |acc, &x| acc | MASK_TABLE[x as usize]) as usize]
}

pub fn is_degenerate(x: u8) -> bool {
    (x != b'A') && (x != b'T') && (x != b'G') && (x != b'C')
}

/// From a codon (ie ANC) return an amino-acid if it's unique
/// or return b'X' if it's not.
pub fn codon_to_amino_acid(codon: [u8; 3]) -> u8 {
    let mut valid_aminos = HashSet::new();
    for a in DNA_TO_AMINO.keys() {
        let nondeg_codon = a.as_bytes();
        if compatible_nucleotides(codon[0], nondeg_codon[0])
            && compatible_nucleotides(codon[1], nondeg_codon[1])
            && compatible_nucleotides(codon[2], nondeg_codon[2])
        {
            valid_aminos.insert(DNA_TO_AMINO[a]);
        }
    }

    if valid_aminos.len() == 1 {
        valid_aminos.drain().next().unwrap()
    } else {
        b'X'
    }
}

pub fn intersect_nucleotides(x: u8, y: u8) -> u8 {
    static MASK_TABLE: [u8; 256] = {
        let mut table = [0; 256];
        table[b'A' as usize] = 0b00000001;
        table[b'C' as usize] = 0b00000010;
        table[b'G' as usize] = 0b00000100;
        table[b'T' as usize] = 0b00001000;
        table[b'N' as usize] = 0b00001111;
        table[b'R' as usize] = 0b00000101; // A or G
        table[b'Y' as usize] = 0b00001010; // T or C
        table[b'S' as usize] = 0b00000110; // G or C
        table[b'W' as usize] = 0b00001001; // A or T
        table[b'K' as usize] = 0b00001100; // G or T
        table[b'M' as usize] = 0b00000011; // A or C
        table[b'B' as usize] = 0b00001110; // C/G/T
        table[b'D' as usize] = 0b00001101; // A/G/T
        table[b'H' as usize] = 0b00001011; // A/C/T
        table[b'V' as usize] = 0b00000111; // A/C/G
        table
    };
    MASK_TABLE[x as usize] & MASK_TABLE[y as usize]
}

pub fn degenerate_dna_to_vec(x: u8) -> Vec<usize> {
    match x {
        b'A' => vec![0],
        b'T' => vec![3],
        b'G' => vec![2],
        b'C' => vec![1],
        b'N' => vec![0, 1, 2, 3],
        b'R' => vec![0, 2],
        b'Y' => vec![1, 3],
        b'S' => vec![1, 2],
        b'W' => vec![0, 3],
        b'K' => vec![2, 3],
        b'M' => vec![0, 1],
        b'B' => vec![1, 2, 3],
        b'D' => vec![0, 2, 3],
        b'H' => vec![0, 1, 3],
        b'V' => vec![0, 1, 2],
        _ => panic!("Wrong character in dna sequence."),
    }
}

pub fn compatible_nucleotides(x: u8, y: u8) -> bool {
    intersect_nucleotides(x, y) != 0
}

pub static NUCLEOTIDES_INV: phf::Map<u8, usize> = phf_map! {
    b'A' => 0, b'T' => 3, b'G' => 2, b'C' => 1, b'N' => 4,
    b'R' => 5, b'Y' => 6, b'S' => 7, b'W' => 8, b'K' => 9,
    b'M' => 10, b'B' => 11, b'D' => 12, b'H' => 13, b'V' => 14,
};

static COMPLEMENT: phf::Map<u8, u8> = phf_map! {
    b'A' => b'T', b'T' => b'A', b'G' => b'C', b'C' => b'G', b'N' => b'N',
    b'R' => b'Y', b'Y' => b'R', b'S' => b'S', b'W' => b'W', b'K' => b'M',
    b'M' => b'K', b'B' => b'V', b'D' => b'H', b'H' => b'D', b'V' => b'B',

};

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

impl fmt::Debug for Dna {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dna [{}]", String::from_utf8_lossy(&self.seq))
    }
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

impl fmt::Debug for AminoAcid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Amino-Acid [{} start:{}  end:{}]",
            String::from_utf8_lossy(&self.translate().unwrap().seq),
            self.start,
            self.end
        )
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

    /// for a sequence of length 2, return the matrix idx
    /// i.e 4*nucleotide[0] + nucleotide[1]
    pub fn to_matrix_idx(&self) -> Vec<usize> {
        debug_assert!(self.len() == 2);
        vec![4 * nucleotides_inv(self.seq[0]) + nucleotides_inv(self.seq[1])]
    }

    pub fn from_matrix_idx(idx: usize) -> Dna {
        Dna {
            seq: vec![NUCLEOTIDES[idx / 4], NUCLEOTIDES[idx % 4]],
        }
    }

    /// return all possible extremities for a given sequence
    /// the extremities are encoded in the index form for the 16x16 likelihood matrix
    /// hence (4*x[-2] +  x[-1], 4*x[x.len()-2] + x[x.len()-1])
    pub fn valid_extremities(&self) -> Vec<(usize, usize)> {
        let mut v = vec![];
        for idx_left in 0..16 {
            let mut seq = Dna::from_matrix_idx(idx_left);
            seq.extend(self);
            let idx_right = seq
                .extract_subsequence(seq.len() - 2, seq.len())
                .to_matrix_idx();
            debug_assert!(idx_right.is_empty());
            v.push((idx_left, idx_right[0]));
        }
        v
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl Dna {
    fn __repr__(&self) -> String {
        String::from("Dna(") + &self.get_string() + &String::from(")")
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
            start: 0,
            end: 0,
        })
    }

    pub fn to_codons(&self) -> Result<AminoAcid> {
        if self.seq.len() % 3 != 0 {
            return Err(anyhow!("Translation not possible, invalid length."))?;
        }

        let amino_sequence: Vec<u8> = self
            .seq
            .chunks(3)
            .map(|codon| {
                128 + (16 * nucleotides_inv(codon[0])
                    + 4 * nucleotides_inv(codon[1])
                    + nucleotides_inv(codon[2])) as u8
            })
            .collect();
        Ok(AminoAcid {
            seq: amino_sequence,
            start: 0,
            end: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.seq.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }

    pub fn extended(&self, other: &Dna) -> Dna {
        let mut s = self.clone();
        s.extend(other);
        s
    }

    pub fn extend(&mut self, dna: &Dna) {
        self.seq.extend(dna.seq.iter());
    }

    pub fn reverse(&mut self) {
        self.seq.reverse();
    }

    /// Return all possible variant of a dna sequence
    /// ANT -> [AAT, ACT, AGT, ATT], can be huge
    pub fn to_dnas(&self) -> Vec<Dna> {
        let mut all_seqs = vec![Dna::new()];
        for a in &self.seq {
            let mut new_seqs = vec![];
            for b in degenerate_dna_to_vec(*a) {
                for seq in &all_seqs {
                    let mut new_seq = seq.clone();
                    new_seq.seq.push(NUCLEOTIDES[b]);
                    new_seqs.push(new_seq);
                }
            }
            all_seqs = new_seqs;
        }
        all_seqs
    }
}

impl Dna {
    pub fn hamming_distance(&self, d: &Dna) -> usize {
        self.seq
            .iter()
            .zip(&d.seq)
            .map(|(&x, &y)| if compatible_nucleotides(x, y) { 0 } else { 1 })
            .sum()
    }

    pub fn count_differences(&self, template: &Dna) -> usize {
        self.hamming_distance(template)
    }

    // also works with slices
    pub fn hamming_distance_u8slice(&self, d: &[u8]) -> usize {
        self.seq
            .iter()
            .zip(d)
            .map(|(&x, &y)| if compatible_nucleotides(x, y) { 0 } else { 1 })
            .sum()
    }

    // also work with index of usize (A,C,G,T => 0,1,2,3)
    pub fn hamming_distance_index_slice(&self, d: &[usize; 3], start: usize, end: usize) -> usize {
        debug_assert!(start + end <= 3);
        if start + end == 3 {
            return 0;
        }
        debug_assert!(self.len() == 3 - start - end);
        self.seq
            .iter()
            .zip(d[start..3 - end].iter())
            .map(|(&x, &y)| {
                if compatible_nucleotides(x, NUCLEOTIDES[y]) {
                    0
                } else {
                    1
                }
            })
            .sum()
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

    // pub fn likelihood(&self, transition_matrix: &Array2<f64>, _first_nt_bias: &Array1<f64>) -> f64 {
    //     let mut proba = 1.;
    //     for ii in 1..self.len() {
    //         proba *= transition_matrix[[
    //             nucleotides_inv(self.seq[ii - 1]),
    //             nucleotides_inv(self.seq[ii]),
    //         ]];
    //     }
    //     proba
    // }

    // pub fn update_transition_matrix(&self, tm: &mut Array2<f64>, likelihood: f64) {
    //     for ii in 1..self.len() {
    //         tm[[
    //             nucleotides_inv(self.seq[ii - 1]),
    //             nucleotides_inv(self.seq[ii]),
    //         ]] += likelihood
    //     }
    // }

    pub fn extract_subsequence(&self, start: usize, end: usize) -> Dna {
        // Return dna[start:end]
        Dna {
            seq: self.seq[start..end].to_vec(),
        }
    }

    /// Return dna[start:end] but padded with N if `start < 0` or `end >= dna.len()`
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
}

impl AminoAcid {
    pub fn translate(&self) -> Result<AminoAcid> {
        Ok(AminoAcid {
            seq: self
                .seq
                .clone()
                .into_iter()
                .map(|x| {
                    if x <= b'Z' {
                        x
                    } else {
                        let y = x - 128;
                        DNA_TO_AMINO[std::str::from_utf8(&[
                            NUCLEOTIDES[(y / 16) as usize],
                            NUCLEOTIDES[((y / 4) % 4) as usize],
                            NUCLEOTIDES[(y % 4) as usize],
                        ])
                        .expect("Problem with the value stored in the amino-acid")]
                    }
                })
                .collect(),
            start: self.start,
            end: self.end,
        })
    }

    pub fn count_differences(&self, template: &Dna) -> usize {
        self.to_degen_cod_seq().count_differences(template)
    }

    /// For a full amino-acid sequence (start & end are 0),
    /// add a DNA sequence before it.
    pub fn append_to_dna_in_frame(&self, seq: &Dna) -> AminoAcid {
        debug_assert!(self.start == 0 && self.end == 0);
        let mut pre = seq
            .extract_subsequence(seq.len() % 3, seq.len())
            .to_codons()
            .unwrap()
            .seq;
        if seq.len() % 3 != 0 {
            pre.insert(0, b'X');
        }
        AminoAcid {
            seq: pre.iter().chain(&self.seq).copied().collect(),
            start: (3 - (seq.len() % 3)) % 3,
            end: 0,
        }
    }

    /// For a full amino-acid sequence (start & end are 0),
    /// add a DNA sequence after it.
    pub fn extend_with_dna_in_frame(&self, seq: &Dna) -> AminoAcid {
        debug_assert!(self.start == 0 && self.end == 0);
        let mut post = seq
            .extract_subsequence(0, seq.len() - seq.len() % 3)
            .to_codons()
            .unwrap()
            .seq;

        if seq.len() % 3 != 0 {
            post.push(b'X');
        }
        AminoAcid {
            seq: self.seq.iter().chain(&post).copied().collect(),
            start: 0,
            end: (3 - (seq.len() % 3)) % 3,
        }
    }

    /// Add two in frame amino-acid sequence
    pub fn extended(&self, seq: &AminoAcid) -> AminoAcid {
        debug_assert!(seq.start == 0 && seq.end == 0 && self.start == 0 && self.end == 0);
        AminoAcid {
            seq: self.seq.iter().chain(&seq.seq).copied().collect(),
            start: 0,
            end: 0,
        }
    }

    /// Extract subsequence (in dna indexing) from the aa sequence
    pub fn extract_subsequence(&self, start: usize, end: usize) -> AminoAcid {
        debug_assert!(end <= 3 * self.seq.len());

        let shift_start = start + self.start;
        let shift_end = end + self.start;

        let aa_start = shift_start / 3;
        // we check where is the last element, divide by 3 then add 1.
        let aa_end = (shift_end + 3 - 1) / 3;

        let new_codons = self.seq[aa_start..aa_end].to_vec();
        AminoAcid {
            seq: new_codons,
            start: shift_start % 3,
            end: 3 * (aa_end) - shift_end,
        }
    }

    fn extract_padded_subsequence(&self, start: i64, end: i64) -> AminoAcid {
        let mut result = self.seq.clone();
        let mut shift = 0;

        let cpos = start + self.start as i64;
        let dpos = cpos.div_euclid(3) * 3;
        let hpos = end + self.start as i64;
        let epos = if hpos.rem_euclid(3) == 0 {
            hpos
        } else {
            hpos.div_euclid(3) * 3 + 3
        };
        let gpos = self.seq.len() as i64;

        if dpos < 0 {
            // pad left
            let new_amino_left = ((-dpos) as usize) / 3;
            let mut left_side = vec![b'X'; new_amino_left];
            left_side.extend_from_slice(&result);
            result = left_side.clone();
            shift = -dpos;
        }
        if epos > gpos {
            // pad right
            let new_amino_right = ((epos - gpos) / 3) as usize;
            result.extend_from_slice(&vec![b'X'; new_amino_right]);
        }

        let aa_start = ((dpos + shift) / 3) as usize;
        let aa_end = ((epos + shift) / 3) as usize;

        AminoAcid {
            seq: result[aa_start..aa_end].to_vec(),
            start: (cpos - dpos) as usize,
            end: (epos - hpos) as usize,
        }
    }

    pub fn to_degen_cod_seq(&self) -> DegenerateCodonSequence {
        DegenerateCodonSequence::from_aminoacid(&self.clone())
    }

    pub fn is_empty(&self) -> bool {
        self.seq.len() == 0 || (self.seq.len() == 1 && (self.start + self.end == 3))
    }

    pub fn to_dna(&self) -> Dna {
        let seq: Vec<_> = self
            .seq
            .iter()
            .flat_map(|x| amino_to_dna_lossy(*x))
            .collect();
        Dna {
            seq: seq[self.start..seq.len() - self.end].to_vec(),
        }
    }

    pub fn reverse(&mut self) {
        self.seq = self.seq.clone().into_iter().rev().collect();
        std::mem::swap(&mut self.start, &mut self.end);
    }

    pub fn from_string(s: &str) -> Result<AminoAcid> {
        for &byte in s.as_bytes() {
            if !AMINOACIDS.contains(&byte) {
                // Handle the error if the byte is not in the map
                return Err(anyhow!(format!("Invalid byte: {}", byte)));
            }
        }

        return Ok(AminoAcid {
            seq: s.as_bytes().to_vec(),
            start: 0,
            end: 0,
        });
    }

    pub fn to_dnas(self) -> Vec<Dna> {
        let mut all_nts = vec![Dna::new()];
        for amino in self.seq {
            let codons = DegenerateCodon::from_amino(amino);
            let mut new_combinations = Vec::new();

            for cod in codons.triplets {
                for nt in &all_nts {
                    let mut new_seq = nt.seq.clone();
                    new_seq.extend(cod.iter().map(|&x| NUCLEOTIDES[x]));
                    new_combinations.push(Dna { seq: new_seq });
                }
            }

            all_nts = new_combinations;
        }
        all_nts
    }
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl AminoAcid {
    fn __repr__(&self) -> String {
        String::from("AminoAcid(") + &self.get_string() + &String::from(")")
    }

    #[staticmethod]
    #[pyo3(name = "from_string")]
    pub fn py_from_string(s: &str) -> Result<AminoAcid> {
        AminoAcid::from_string(s)
    }

    fn get_string(&self) -> String {
        String::from_utf8_lossy(&self.seq).to_string()
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

// pyo3 boiler code
#[cfg(feature = "py_binds")]
#[pymethods]
impl AminoAcid {
    #[new]
    #[pyo3(signature = (sequence = ""))]
    pub fn py_new(sequence: &str) -> Result<AminoAcid> {
        AminoAcid::from_string(sequence)
    }
}

// Boiler code, needed for pyo3
// pyo3 boiler code
#[cfg(feature = "py_binds")]
#[pymethods]
impl DnaLike {
    #[new]
    #[pyo3(signature = (sequence = ""))]
    pub fn py_new(sequence: &str) -> Result<DnaLike> {
        DnaLike::from_string(sequence, "dna")
    }

    #[staticmethod]
    #[pyo3(name = "from_string", signature = (s = "", sequence_type = "dna"))]
    /// Load a DnaLike object from either an amino-acid or a nucleotide sequence
    /// Sequence type is either "dna" or "aa" by default try "dna"
    pub fn py_from_string(s: &str, sequence_type: &str) -> Result<DnaLike> {
        DnaLike::from_string(s, sequence_type)
    }

    #[staticmethod]
    #[pyo3(name = "from_amino_acid")]
    pub fn py_from_amino_acid(seq: AminoAcid) -> PyResult<DnaLike> {
        Ok(DnaLike {
            inner: DnaLikeEnum::from_amino_acid(seq),
        })
    }

    #[staticmethod]
    #[pyo3(name = "from_amino_dna")]
    pub fn py_from_dna(seq: Dna) -> PyResult<DnaLike> {
        Ok(DnaLike {
            inner: DnaLikeEnum::from_dna(seq),
        })
    }
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl DnaLike {
    // pure python functions
    pub fn __repr__(&self) -> String {
        match &self.inner {
            DnaLikeEnum::Known(s) | DnaLikeEnum::Ambiguous(s) => s.__repr__(),
            DnaLikeEnum::Protein(s) => s.__repr__(),
        }
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl DnaLike {
    pub fn to_dna(&self) -> Dna {
        self.inner.to_dna()
    }
    pub fn get_string(&self) -> String {
        self.inner.get_string()
    }

    pub fn translate(&self) -> Result<AminoAcid> {
        self.inner.translate()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn is_protein(&self) -> bool {
        self.inner.is_protein()
    }

    pub fn is_ambiguous(&self) -> bool {
        match self.inner {
            DnaLikeEnum::Known(_) => false,
            DnaLikeEnum::Ambiguous(_) | DnaLikeEnum::Protein(_) => true,
        }
    }

    pub fn sequence_type(&self) -> SequenceType {
        if self.inner.is_protein() {
            SequenceType::Protein
        } else {
            SequenceType::Dna
        }
    }

    // pub fn valid_extremities(&self) -> Vec<(usize, usize)> {
    //     self.inner.valid_extremities()
    // }

    pub fn reverse(&mut self) {
        self.inner.reverse();
    }
}

impl DnaLike {
    pub fn extended_in_frame(&self, seq: &DnaLike) -> DnaLike {
        DnaLike {
            inner: self.inner.extended_in_frame(&seq.inner),
        }
    }

    pub fn to_dnas(&self) -> Vec<Dna> {
        self.inner.clone().to_dnas()
    }

    pub fn from_amino_acid(seq: AminoAcid) -> DnaLike {
        DnaLike {
            inner: DnaLikeEnum::from_amino_acid(seq),
        }
    }
    pub fn from_dna(seq: Dna) -> DnaLike {
        DnaLike {
            inner: DnaLikeEnum::from_dna(seq),
        }
    }

    pub fn from_string(s: &str, sequence_type: &str) -> Result<DnaLike> {
        match sequence_type {
            "dna" => Ok(DnaLike::from_dna(Dna::from_string(s)?)),
            "aa" => Ok(DnaLike::from_amino_acid(AminoAcid::from_string(s)?)),
            _ => Err(anyhow!(
                "Wrong `sequence_type`, can be either \"dna\" (nucleotides) or \"aa\" (amino-acid)"
            )),
        }
    }

    pub fn to_matrix_idx(&self) -> Vec<usize> {
        self.inner.to_matrix_idx()
    }

    // pub fn extend(&mut self, other: DnaLike) {
    //     self.inner.extend(&other.inner)
    // }

    pub fn extended(&self, other: &DnaLike) -> DegenerateCodonSequence {
        self.inner.extended(&other.inner)
    }

    pub fn extract_subsequence(&self, start: usize, end: usize) -> DnaLike {
        DnaLike {
            inner: self.inner.extract_subsequence(start, end),
        }
    }

    pub fn extract_padded_subsequence(&self, start: i64, end: i64) -> DnaLike {
        DnaLike {
            inner: self.inner.extract_padded_subsequence(start, end),
        }
    }

    // pub fn get_nucleotide_index(&self, pos: i64) -> usize {
    //     self.inner.get_nucleotide_index(pos)
    // }

    pub fn align_left_right(
        sleft: DnaLike,
        sright: DnaLike,
        align_params: &AlignmentParameters,
    ) -> Alignment {
        DnaLikeEnum::align_left_right(&sleft.into(), &sright.into(), align_params)
    }

    pub fn v_alignment(
        vgene: &Dna,
        sequence: DnaLike,
        align_params: &AlignmentParameters,
    ) -> Option<Alignment> {
        DnaLikeEnum::v_alignment(vgene, &sequence.into(), align_params)
    }

    pub fn count_differences(&self, template: &Dna) -> usize {
        self.inner.count_differences(template)
    }
}
