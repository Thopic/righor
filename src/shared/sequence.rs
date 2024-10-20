use crate::shared::utils::mod_euclid;
/// Contains the basic struct and function for loading and aligning sequences
use crate::shared::AlignmentParameters;
use anyhow::{anyhow, Result};
use bio::alignment::{pairwise, Alignment};
use std::collections::HashSet;

use phf::phf_map;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

/////////////////////////////////////
// Structures
/////////////////////////////////////

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Clone, Debug, Copy, Default)]
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
    Known(Dna),                       // known Dna (A/T/G/C)
    Ambiguous(Dna),                   // degenerate Dna (contains degenerate nucleotides N/H...)
    Protein(DegenerateCodonSequence), // reverse-translated amino-acid sequence
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
/// Partially defined dna sequence created from an amino-acid sequence
/// Example: NHLRH, 1, 2 corresponds to (among many others) .atcacctaagacat
pub struct DegenerateCodonSequence {
    // List of codons
    pub codons: Vec<DegenerateCodon>,
    // the start of the actual nucleotide sequence (potentially mid-codon)
    // belong to  0..=2
    pub codon_start: usize,
    // the end of the actual nucleotide sequence (potentially mid-codon)
    // belong to 0..=2 (go in reverse, 0 full aa encoded, 2, just the first aa)
    pub codon_end: usize,
}

/// Dna sequence (for A/T/G/C, but also used internally for degenerate nucleotides)
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
pub struct Dna {
    pub seq: Vec<u8>,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, PartialEq, Eq)]
pub struct AminoAcid {
    pub seq: Vec<u8>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
/// Define a degenerated codon, ie a  vec of triplets of nucleotides
/// nucleotides are stored as 0,1,2,3
pub struct DegenerateCodon {
    pub triplets: Vec<[usize; 3]>,
}

/////////////////////////////////////
// Implementations
/////////////////////////////////////

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
            DnaLikeEnum::Protein(s) => s.len(),
        }
    }

    // for a sequence of length 2, return all the possible matrix idx
    // i.e 4*nucleotide[0] + nucleotide[1]
    pub fn to_matrix_idx(&self) -> Vec<usize> {
        match self {
            DnaLikeEnum::Known(s) | DnaLikeEnum::Ambiguous(s) => s.to_matrix_idx(),
            DnaLikeEnum::Protein(s) => s.to_matrix_idx(),
        }
    }

    pub fn reverse(&mut self) {
        match self {
            DnaLikeEnum::Known(s) | DnaLikeEnum::Ambiguous(s) => s.reverse(),
            DnaLikeEnum::Protein(s) => s.reverse(),
        }
    }

    /// Send the DnaLikeEnum to a Dna object. Can be lossy
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
        DnaLikeEnum::Protein(DegenerateCodonSequence::from_aminoacid(seq))
    }

    // TODO: this can be coded better (but probably doesn't matter)
    pub fn extend(&mut self, other: &DnaLikeEnum) {
        *self = match (self.clone(), other) {
            // Known cases
            (DnaLikeEnum::Known(mut self_dna), DnaLikeEnum::Known(other_dna)) => {
                self_dna.extend(other_dna);
                DnaLikeEnum::Known(self_dna.clone())
            }
            (DnaLikeEnum::Known(self_dna), DnaLikeEnum::Ambiguous(other_dna)) => {
                let mut new_dna = self_dna.clone();
                new_dna.extend(other_dna);
                DnaLikeEnum::Ambiguous(new_dna)
            }
            (DnaLikeEnum::Known(self_dna), DnaLikeEnum::Protein(other_protein)) => {
                let mut new_protein = other_protein.clone();
                new_protein.append_to_dna(&self_dna);
                DnaLikeEnum::Protein(new_protein)
            }
            // Ambiguous cases
            (DnaLikeEnum::Ambiguous(mut self_dna), DnaLikeEnum::Known(other_dna)) => {
                self_dna.extend(other_dna);
                DnaLikeEnum::Ambiguous(self_dna.clone())
            }
            (DnaLikeEnum::Ambiguous(mut self_dna), DnaLikeEnum::Ambiguous(other_dna)) => {
                self_dna.extend(other_dna);
                DnaLikeEnum::Ambiguous(self_dna.clone())
            }
            (DnaLikeEnum::Ambiguous(self_dna), DnaLikeEnum::Protein(other_protein)) => {
                let mut new_protein = other_protein.clone();
                new_protein.append_to_dna(&self_dna);
                DnaLikeEnum::Protein(new_protein)
            }
            // Protein cases
            (DnaLikeEnum::Protein(mut self_protein), DnaLikeEnum::Known(other_dna)) => {
                self_protein.extend_dna(other_dna);
                DnaLikeEnum::Protein(self_protein.clone())
            }
            (DnaLikeEnum::Protein(mut self_protein), DnaLikeEnum::Ambiguous(other_dna)) => {
                self_protein.extend_dna(other_dna);
                DnaLikeEnum::Protein(self_protein.clone())
            }
            (DnaLikeEnum::Protein(mut self_protein), DnaLikeEnum::Protein(other_protein)) => {
                self_protein.extend(other_protein);
                DnaLikeEnum::Protein(self_protein.clone())
            }
        };
    }

    pub fn extract_subsequence(&self, start: usize, end: usize) -> DnaLikeEnum {
        match self {
            DnaLikeEnum::Known(s) => DnaLikeEnum::Known(s.extract_subsequence(start, end)),
            DnaLikeEnum::Ambiguous(s) => DnaLikeEnum::Ambiguous(s.extract_subsequence(start, end)),
            DnaLikeEnum::Protein(s) => DnaLikeEnum::Protein(s.extract_subsequence(start, end)),
        }
    }

    /// Return dna[start:end] but padded with N if start < 0 or end >= dna.len()
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
    // pub fn extremities(&self) -> Vec<(usize, usize)> {
    //     match self {
    //         DnaLikeEnum::Known(s) => s.extremities(),
    //         DnaLikeEnum::DegenerateDNA(s) => s.extremities(),
    //         DnaLikeEnum::Protein(s) => s.extremities(),
    //     }
    // }
}

impl DegenerateCodon {
    // Return a matrix A(τ, σ), where τ is the last nucleotide of the previous codon and
    // σ is the last nucleotide
    // pub fn likelihood_matrix(&self, transition_matrix_undefined: &Array2<f64>) -> Matrix4<f64> {
    //     let mut matrix = Matrix4::new(
    //         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    //     );

    //     for x in self.triplets.iter() {
    //         for ii in 0..4 {
    //             for jj in 0..4 {
    //                 if compatible_nucleotides(NUCLEOTIDES[x[2]], NUCLEOTIDES[jj]) {
    //                     matrix[(ii, jj)] += transition_matrix_undefined[[ii, x[0]]]
    //                         * transition_matrix_undefined[[x[0], x[1]]]
    //                         * transition_matrix_undefined[[x[1], jj]];
    //                 }
    //             }
    //         }
    //     }

    //     matrix
    // }

    pub fn translate(&self) -> u8 {
        let mut b = None;
        for v in self.triplets.iter() {
            let codons_utf8 = [NUCLEOTIDES[v[0]], NUCLEOTIDES[v[1]], NUCLEOTIDES[v[2]]];
            let a = codon_to_amino_acid(codons_utf8);
            match b {
                Some(x) if a != x => return b'X', // degenerate
                None => b = Some(a),
                Some(_) => {}
            }
        }
        match b {
            None => panic!("Error with translation of an ExtendedDna object"),
            Some(aa) => aa,
        }
    }

    // /// Return the likelihood vector "probability that the next nucleotide after this codon is X".
    // pub fn likelihood_start(
    //     &self,
    //     start: usize,
    //     transition_matrix_undefined: &Array2<f64>,
    // ) -> Vector4<f64> {
    //     let mut vector = Vector4::new(0., 0., 0., 0.);
    //     if start == 0 {
    //         for x in &self.triplets {
    //             for ii in 0..4 {
    //                 vector[ii] += transition_matrix_undefined[[x[0], x[1]]]
    //                     * transition_matrix_undefined[[x[1], x[2]]]
    //                     * transition_matrix_undefined[[x[2], ii]];
    //             }
    //         }
    //     } else if start == 1 {
    //         for x in &self.triplets {
    //             for ii in 0..4 {
    //                 vector[ii] += transition_matrix_undefined[[x[1], x[2]]]
    //                     * transition_matrix_undefined[[x[2], ii]];
    //             }
    //         }
    //     } else if start == 2 {
    //         for x in &self.triplets {
    //             for ii in 0..4 {
    //                 vector[ii] += transition_matrix_undefined[[x[2], ii]];
    //             }
    //         }
    //     }
    //     vector
    // }

    // /// Return the likelihood vector p(ii, last)
    // pub fn likelihood_end(
    //     &self,
    //     end: usize,
    //     transition_matrix_undefined: &Array2<f64>,
    // ) -> Vector4<f64> {
    //     let mut vector = Vector4::new(0., 0., 0., 0.);
    //     if end == 0 {
    //         for x in &self.triplets {
    //             for ii in 0..4 {
    //                 vector[ii] += transition_matrix_undefined[[ii, x[0]]]
    //                     * transition_matrix_undefined[[x[0], x[1]]]
    //                     * transition_matrix_undefined[[x[1], x[2]]];
    //             }
    //         }
    //     } else if end == 1 {
    //         for x in &self.triplets {
    //             for ii in 0..4 {
    //                 vector[ii] += transition_matrix_undefined[[ii, x[0]]]
    //                     * transition_matrix_undefined[[x[0], x[1]]];
    //             }
    //         }
    //     } else if end == 2 {
    //         for x in &self.triplets {
    //             for ii in 0..4 {
    //                 vector[ii] += transition_matrix_undefined[[ii, x[0]]];
    //             }
    //         }
    //     }
    //     vector
    // }

    // /// Return the likelihood within one codon (special case)
    // pub fn likelihood_one_codon(
    //     &self,
    //     start: usize,
    //     end: usize,
    //     transition_matrix: &Array2<f64>,
    // ) -> f64 {
    //     // this should depend of the first nucleotide.
    //     match (start, end) {
    //         (0, 1) => self
    //             .triplets
    //             .iter()
    //             .map(|x| transition_matrix[[x[0], x[1]]])
    //             .sum(),
    //         (1, 0) => self
    //             .triplets
    //             .iter()
    //             .map(|x| transition_matrix[[x[1], x[2]]])
    //             .sum(),
    //         (0, 0) => self
    //             .triplets
    //             .iter()
    //             .map(|x| transition_matrix[[x[0], x[1]]] * transition_matrix[[x[1], x[2]]])
    //             .sum(),
    //         (_, _) => panic!("Wrong values in codon likelihood computation (likelihood_one_codon)"),
    //     }
    // }

    /// Return all the possible codons (64), in a fixed order
    pub fn all_codons() -> Vec<[usize; 3]> {
        let mut v = vec![];
        for n1 in 0..4 {
            for n2 in 0..4 {
                for n3 in 0..4 {
                    v.push([n1, n2, n3])
                }
            }
        }
        v
    }

    /// Return the index of a given nucleotide triplet. all_codons()[get_index(x)] = x
    pub fn get_index(cod: &[usize; 3]) -> usize {
        cod[2] + 4 * cod[1] + 16 * cod[0]
    }

    pub fn from_amino(x: u8) -> DegenerateCodon {
        DegenerateCodon {
            triplets: match x {
                b'A' => vec![[2, 1, 3], [2, 1, 1], [2, 1, 0], [2, 1, 2]], // GCN
                b'C' => vec![[3, 2, 3], [3, 2, 1]],                       // TGY
                b'D' => vec![[2, 0, 3], [2, 0, 1]],                       // GAY
                b'E' => vec![[2, 0, 0], [2, 0, 2]],                       // GAR
                b'F' => vec![[3, 3, 3], [3, 3, 1]],                       // TTY
                b'G' => vec![[2, 2, 3], [2, 2, 1], [2, 2, 0], [2, 2, 2]], // GGN
                b'H' => vec![[1, 0, 3], [1, 0, 1]],                       // CAY
                b'I' => vec![[0, 3, 3], [0, 3, 1], [0, 3, 0]],            // ATH
                b'L' => vec![
                    [1, 3, 3],
                    [1, 3, 1],
                    [1, 3, 0],
                    [1, 3, 2],
                    [3, 3, 0],
                    [3, 3, 2],
                ], // CTN, TTR
                b'K' => vec![[0, 0, 0], [0, 0, 2]],                       // AAR
                b'M' => vec![[0, 3, 2]],                                  // ATG
                b'N' => vec![[0, 0, 3], [0, 0, 1]],                       //AAY
                b'P' => vec![[1, 1, 3], [1, 1, 1], [1, 1, 0], [1, 1, 2]], // CCN
                b'Q' => vec![[1, 0, 0], [1, 0, 2]],                       // CAR
                b'R' => vec![
                    [1, 2, 3],
                    [1, 2, 1],
                    [1, 2, 0],
                    [1, 2, 2],
                    [0, 2, 0],
                    [0, 2, 2],
                ], // CGN, AGR
                b'S' => vec![
                    [3, 1, 3],
                    [3, 1, 1],
                    [3, 1, 0],
                    [3, 1, 2],
                    [0, 2, 3],
                    [0, 2, 1],
                ], // TCN, AGS
                b'T' => vec![[0, 1, 3], [0, 1, 1], [0, 1, 0], [0, 1, 2]], // ACN
                b'V' => vec![[2, 3, 3], [2, 3, 1], [2, 3, 0], [2, 3, 2]], // GTN
                b'W' => vec![[3, 2, 2]],                                  // TGG
                b'Y' => vec![[3, 0, 3], [3, 0, 1]],                       // TAY
                b'*' => vec![[3, 0, 0], [3, 0, 2], [3, 2, 0]],            // TAR, TGA
                b'X' => vec![
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [0, 0, 3],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 1, 2],
                    [0, 1, 3],
                    [0, 2, 0],
                    [0, 2, 1],
                    [0, 2, 2],
                    [0, 2, 3],
                    [0, 3, 0],
                    [0, 3, 1],
                    [0, 3, 2],
                    [0, 3, 3],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 0, 2],
                    [1, 0, 3],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 2],
                    [1, 1, 3],
                    [1, 2, 0],
                    [1, 2, 1],
                    [1, 2, 2],
                    [1, 2, 3],
                    [1, 3, 0],
                    [1, 3, 1],
                    [1, 3, 2],
                    [1, 3, 3],
                    [2, 0, 0],
                    [2, 0, 1],
                    [2, 0, 2],
                    [2, 0, 3],
                    [2, 1, 0],
                    [2, 1, 1],
                    [2, 1, 2],
                    [2, 1, 3],
                    [2, 2, 0],
                    [2, 2, 1],
                    [2, 2, 2],
                    [2, 2, 3],
                    [2, 3, 0],
                    [2, 3, 1],
                    [2, 3, 2],
                    [2, 3, 3],
                    [3, 0, 0],
                    [3, 0, 1],
                    [3, 0, 2],
                    [3, 0, 3],
                    [3, 1, 0],
                    [3, 1, 1],
                    [3, 1, 2],
                    [3, 1, 3],
                    [3, 2, 0],
                    [3, 2, 1],
                    [3, 2, 2],
                    [3, 2, 3],
                    [3, 3, 0],
                    [3, 3, 1],
                    [3, 3, 2],
                    [3, 3, 3],
                ], // NNN
                _ => panic!("Wrong amino-acid sequence"),
            },
        }
    }

    pub fn from_dna(x: &Dna) -> DegenerateCodon {
        debug_assert!(x.len() == 3);
        DegenerateCodon {
            triplets: vec![[
                nucleotides_inv(x.seq[0]),
                nucleotides_inv(x.seq[1]),
                nucleotides_inv(x.seq[2]),
            ]],
        }
    }

    // lossy method
    pub fn to_dna(&self) -> Dna {
        Dna {
            seq: vec![
                degenerate_nucleotide(
                    &self
                        .triplets
                        .iter()
                        .map(|x| NUCLEOTIDES[x[0]])
                        .collect::<Vec<u8>>(),
                ),
                degenerate_nucleotide(
                    &self
                        .triplets
                        .iter()
                        .map(|x| NUCLEOTIDES[x[1]])
                        .collect::<Vec<u8>>(),
                ),
                degenerate_nucleotide(
                    &self
                        .triplets
                        .iter()
                        .map(|x| NUCLEOTIDES[x[2]])
                        .collect::<Vec<u8>>(),
                ),
            ],
        }
    }

    pub fn from_u8(x: &[u8]) -> DegenerateCodon {
        debug_assert!(x.len() == 3);
        DegenerateCodon {
            triplets: vec![[
                nucleotides_inv(x[0]),
                nucleotides_inv(x[1]),
                nucleotides_inv(x[2]),
            ]],
        }
    }

    /// Return a new codon, with the end nucleotides at the end of
    /// the og codon replaced with the dna sequence
    pub fn end_replace(&self, end: usize, seq: &Dna) -> DegenerateCodon {
        debug_assert!(seq.len() == end);
        debug_assert!(end == 1 || end == 2 || end == 0);
        DegenerateCodon {
            triplets: match end {
                1 => self
                    .triplets
                    .iter()
                    .map(|x| [x[0], x[1], nucleotides_inv(seq.seq[0])])
                    .collect(),
                2 => self
                    .triplets
                    .iter()
                    .map(|x| {
                        [
                            x[0],
                            nucleotides_inv(seq.seq[0]),
                            nucleotides_inv(seq.seq[1]),
                        ]
                    })
                    .collect(),
                0 => self.triplets.clone(),
                _ => panic!("Wrong end codon value"),
            },
        }
    }

    /// Replace the start nucleotides at the end of the codon with the dna sequence
    pub fn start_replace(&self, start: usize, seq: &Dna) -> DegenerateCodon {
        debug_assert!(seq.len() == start);
        debug_assert!(start == 1 || start == 2 || start == 0);
        DegenerateCodon {
            triplets: match start {
                1 => self
                    .triplets
                    .iter()
                    .map(|x| [nucleotides_inv(seq.seq[0]), x[1], x[2]])
                    .collect(),
                2 => self
                    .triplets
                    .iter()
                    .map(|x| {
                        [
                            nucleotides_inv(seq.seq[0]),
                            nucleotides_inv(seq.seq[1]),
                            x[2],
                        ]
                    })
                    .collect(),
                0 => self.triplets.clone(),
                _ => panic!("Wrong start codon value"),
            },
        }
    }

    /// Return the hamming distance between the codon and a dna sequence
    /// of length 3 - start - end. start + end <= 2
    pub fn hamming_distance(&self, seq: &Dna, start: usize, end: usize) -> usize {
        debug_assert!(seq.len() == 3 - start - end);

        // println!(
        //     "TRUC: {:?}",
        //     self.triplets
        //         .iter()
        //         .map(|x| (x, seq.hamming_distance_index_slice(x, start, end)))
        //         .collect::<Vec<_>>()
        // );

        self.triplets
            .iter()
            .map(|x| seq.hamming_distance_index_slice(x, start, end))
            .min()
            .unwrap()
    }

    pub fn reverse(&self) -> DegenerateCodon {
        DegenerateCodon {
            triplets: self.triplets.iter().map(|x| [x[2], x[1], x[0]]).collect(),
        }
    }
}

impl DegenerateCodonSequence {
    pub fn pad_right(&mut self, n: usize) {
        // X for undefined amino-acid
        self.extend_dna(&Dna { seq: vec![b'N'; n] });
    }

    pub fn pad_left(&mut self, n: usize) {
        // NNN corresponds to all possible amino-acid, so to the bitvec (2^64 - 1)
        self.append_to_dna(&Dna { seq: vec![b'N'; n] });
    }

    /// Make an amino-acid sequence into an "UndefinedDna" sequence
    pub fn from_aminoacid(aa: AminoAcid) -> DegenerateCodonSequence {
        DegenerateCodonSequence {
            codons: aa
                .seq
                .iter()
                .map(|&x| DegenerateCodon::from_amino(x))
                .collect(),
            codon_start: 0,
            codon_end: 0,
        }
    }

    /// Make a (non-degenerate) nucleotide sequence into an "UndefinedDna" sequence
    /// For example ATCG, start = 1 would give [[NAT],[CGN]]
    pub fn from_dna(seq: &Dna, start: usize) -> DegenerateCodonSequence {
        let end = mod_euclid(3 - seq.len() as i64 - start as i64, 3) as usize;
        let mut padded_dna = Dna {
            seq: vec![b'N'; start],
        };
        padded_dna.extend(seq);
        padded_dna.extend(&Dna {
            seq: vec![b'N'; end],
        });

        DegenerateCodonSequence {
            codons: padded_dna.seq[..]
                .chunks(3)
                .map(|x| DegenerateCodon::from_u8(x))
                .collect(),
            codon_start: start,
            codon_end: mod_euclid(3 - seq.len() as i64 - start as i64, 3) as usize,
        }
    }

    pub fn translate(&self) -> Result<AminoAcid> {
        if self.codon_start != 0 || self.codon_end != 0 {
            return Err(anyhow!(
                "Translation not possible, wrong reading frame/length."
            ))?;
        }
        Ok(AminoAcid {
            seq: self.codons.iter().map(|x| x.translate()).collect(),
        })
    }

    pub fn to_matrix_idx(&self) -> Vec<usize> {
        debug_assert!(self.len() == 2);

        let mut result = vec![];

        if self.codon_start == 0 {
            for cod in self.codons[0].triplets.iter() {
                result.push(4 * cod[0] + cod[1]);
            }
        } else if self.codon_start == 1 {
            for cod in self.codons[0].triplets.iter() {
                result.push(4 * cod[1] + cod[2]);
            }
        } else if self.codon_start == 2 {
            for cod1 in self.codons[0].triplets.iter() {
                for cod2 in self.codons[1].triplets.iter() {
                    result.push(4 * cod1[2] + cod2[0]);
                }
            }
        }
        return result;
    }

    //    /// return all possible extremities for a given sequence
    //    /// the extremities are encoded in the index form for the 16x16 likelihood matrix
    //    /// hence (4*x[0] +  x[1], 4*x[x.len()] + x[x.len()+1])
    //    pub fn extremities(&self) -> Vec<usize> {}

    // /// For a sequence of length >= 1 return a vector with
    // /// all possible sequences given the first nucleotide (repr. as 0/1/2/3)
    // pub fn fix_first(&self, n: usize) -> Vec<(u8, DegenerateCodonSequence)> {
    //     debug_assert!(self.len() >= n);

    //     let left_extr = vec![];

    //     for nuc in 0..4 {
    //         let first_codon = DegenerateCodon {
    //             triplets: self.codons[0]
    //                 .triplets
    //                 .iter()
    //                 .filter(|x| x[self.codon_start] == nuc),
    //         };
    //         let mut new_codons = self.codons.clone();
    //         new_codons[0] = first_codon;

    //         left_extr.push(DegenerateCodonSequence {
    //             codons: new_codons,
    //             codon_start: self.codon_start + 1,
    //             codon_end: self.codon_end,
    //         })
    //     }
    // }

    /// For a sequence of length larger than 2, return a vector
    /// with all possible left extremities + the sequence
    /// For ex (AT|GA)CNTA => [(4*0 + 3, ATCNTA), (4*2 + 0, GACNTA)]
    pub fn fix_left(&self) -> Vec<(usize, DegenerateCodonSequence)> {
        debug_assert!(self.len() >= 2);
        let mut left_extr = vec![];

        if self.codon_start == 0 {
            for left_codon in self.codons[0].triplets.iter() {
                let mut new_codons = self.codons.clone();
                new_codons[0] = DegenerateCodon {
                    triplets: vec![*left_codon],
                };
                left_extr.push((
                    4 * left_codon[0] + left_codon[1],
                    DegenerateCodonSequence {
                        codons: new_codons,
                        codon_start: 0,
                        codon_end: self.codon_end,
                    },
                ))
            }
        }

        if self.codon_start == 1 {
            for left_codon in self.codons[0].triplets.iter() {
                let mut new_codons = self.codons.clone();
                new_codons[0] = DegenerateCodon {
                    triplets: vec![*left_codon],
                };
                left_extr.push((
                    4 * left_codon[1] + left_codon[2],
                    DegenerateCodonSequence {
                        codons: new_codons,
                        codon_start: 1,
                        codon_end: self.codon_end,
                    },
                ))
            }
        }

        if self.codon_start == 2 {
            for left_codon in self.codons[0].triplets.iter() {
                for left_2_codon in self.codons[1].triplets.iter() {
                    let mut new_codons = self.codons.clone();
                    new_codons[0] = DegenerateCodon {
                        triplets: vec![*left_codon],
                    };
                    new_codons[1] = DegenerateCodon {
                        triplets: vec![*left_2_codon],
                    };
                    left_extr.push((
                        4 * left_codon[2] + left_2_codon[0],
                        DegenerateCodonSequence {
                            codons: new_codons,
                            codon_start: 2,
                            codon_end: self.codon_end,
                        },
                    ))
                }
            }
        }
        left_extr
    }

    /// For a sequence, return a vector with all possible right extremities
    /// (len+1, len+2) + the sequence
    pub fn fix_right(&self) -> Vec<(usize, DegenerateCodonSequence)> {
        debug_assert!(self.len() >= 2);
        let mut right_extr = vec![];
        let idx_last = self.codons.len() - 1;

        if self.codon_end == 0 {
            // all extremities are possible
            for a in 0..16 {
                right_extr.push((a, self.clone()))
            }
        }
        if self.codon_end == 1 {
            for a in 0..16 {
                let mut sequence = DegenerateCodonSequence {
                    codons: vec![],
                    codon_start: self.codon_start,
                    codon_end: self.codon_end,
                };
                for right_codon in self.codons[idx_last].triplets.iter() {
                    if right_codon[2] == a / 4 {
                        sequence.codons.push(DegenerateCodon {
                            triplets: vec![*right_codon],
                        })
                    }
                }
                right_extr.push((a, sequence));
            }
        }

        if self.codon_end == 2 {
            for a in 0..16 {
                let mut sequence = DegenerateCodonSequence {
                    codons: vec![],
                    codon_start: self.codon_start,
                    codon_end: self.codon_end,
                };
                for right_codon in self.codons[idx_last].triplets.iter() {
                    if 4 * right_codon[1] + right_codon[2] == a {
                        sequence.codons.push(DegenerateCodon {
                            triplets: vec![*right_codon],
                        })
                    }
                }
                right_extr.push((a, sequence));
            }
        }
        right_extr
    }

    /// Return all possible extremities for a given sequence
    /// and the sequence with these extremities on.
    /// Extremities corresponds to 4*s[0] + s[1] and
    /// 4 * s[len] + s[len+1]
    pub fn fix_extremities(&self) -> Vec<(usize, usize, DegenerateCodonSequence)> {
        let mut all_extremities = vec![];
        // first the weird cases where left and right interact
        if self.len() == 0 {
            // identity matrix
            for extr in 0..16 {
                all_extremities.push((extr, extr, self.clone()))
            }
            return all_extremities;
        } else if self.len() == 1 {
            for (left_extr, seq) in self.fix_left() {
                for (right_extr, fixed_seq) in seq.fix_right() {
                    if right_extr / 4 == left_extr % 4 {
                        all_extremities.push((left_extr, right_extr, fixed_seq.clone()))
                    }
                }
            }
            return all_extremities;
        }

        // now the normal situation where left_extr and right_extr are fully separated
        for (left_extr, seq) in self.fix_left() {
            for (right_extr, fixed_seq) in seq.fix_right() {
                all_extremities.push((left_extr, right_extr, fixed_seq.clone()))
            }
        }
        return all_extremities;
    }

    /// lossy process, remove some information about the codon
    pub fn to_dna(&self) -> Dna {
        let sequence = Dna {
            seq: self.codons.iter().flat_map(|aa| aa.to_dna().seq).collect(),
        };
        //        println!("{:?}", sequence);
        sequence.extract_subsequence(self.codon_start, self.len() + self.codon_start)
    }

    /// Extract a subsequence from the dna. No particular checks.
    pub fn extract_subsequence(&self, start: usize, end: usize) -> DegenerateCodonSequence {
        // where to start in the amino-acid sequence

        // example:
        // start = 10, end = 20
        // codon_start = 2, codon_end = 1
        //   <---------------------------------->  : true sequence
        // ....................................... : full stored data
        //  x  x  x  x  x  x  x  x  x  x  x  x  x  : amino-acids
        //             <-------->                  : extracted sequence
        //             ..........                  : stored data for extracted sequence
        //
        debug_assert!(end <= self.len());

        let shift_start = start + self.codon_start;
        let shift_end = end + self.codon_start;

        let aa_start = shift_start / 3;
        // we check where is the last element, divide by 3 then add 1.
        let aa_end = (shift_end + 3 - 1) / 3;
        DegenerateCodonSequence {
            codons: self.codons[aa_start..aa_end].to_vec(),
            codon_start: shift_start % 3,
            codon_end: 3 * (aa_end) - shift_end,
        }
    }

    /// Return dna[start:end] but padded with N if start < 0 or end >= dna.len()
    pub fn extract_padded_subsequence(&self, start: i64, end: i64) -> DegenerateCodonSequence {
        // example:
        // start = -4, end = 17
        // codon_start = 2, codon_end = 0
        //               0    '    '   '
        //               <----------->             : true sequence
        //             ...............             : full stored data
        //              x  x  x  x  x              : amino-acids
        //           <-------------------->        : extracted and padded sequence
        //          ........................       : stored data for padded sequence

        let mut result = self.clone();
        let mut shift = 0;

        if start < 0 {
            result.pad_left(start.unsigned_abs() as usize);
            shift = start.unsigned_abs() as i64;
        }
        if end > self.len() as i64 {
            result.pad_right((end - self.len() as i64) as usize);
        }
        result.extract_subsequence((start + shift) as usize, (end + shift) as usize)
    }

    /// Count the number of difference between the sequence and a template
    /// Hamming distance, choosing the most favorable codon each time
    pub fn count_differences(&self, template: &Dna) -> usize {
        let mut distance = 0;
        let mut current = 0;
        for (ii, cs) in self.codons.iter().enumerate() {
            let start = if ii == 0 { self.codon_start } else { 0 };
            let end = if ii == self.codons.len() - 1 {
                self.codon_end
            } else {
                0
            };

            distance += cs.hamming_distance(
                &Dna {
                    seq: template.seq[current..current + 3 - start - end].to_vec(),
                },
                start,
                end,
            );
            // println!(
            //     "{:?} {:?}",
            //     cs,
            //     Dna {
            //         seq: template.seq[current..current + 3 - start - end]
            //             .iter()
            //             .cloned()
            //             .collect::<Vec<_>>()
            //     }
            //     .to_string()
            // );
            // println!("DIST {}", distance);
            current += 3 - start - end;
        }
        distance
    }

    pub fn len(&self) -> usize {
        3 * self.codons.len() - self.codon_start - self.codon_end
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reverse(&mut self) {
        self.codons = self.codons.iter_mut().map(|x| x.reverse()).collect();
        std::mem::swap(&mut self.codon_start, &mut self.codon_end);
    }

    pub fn extend(&mut self, _dna: &DegenerateCodonSequence) {
        // this is very weird because frame shift, should happend
        unimplemented!("Appending two DegenerateCodonSequence shouldn't happen.")
    }

    /// Add a dna sequence at the end of an DegenerateCodonSequence sequence
    pub fn extend_dna(&mut self, dna: &Dna) {
        //               <---------->xxxxxx
        //             ..............xxxxxx
        //              x  x  x  x  x  x  x
        // Note: this is complicated to implement as we need to deal
        // with all the painful edge cases (empty dna, very short dna)
        // need extensive testing

        // if len(dna) < self.codon_end we can't fully complete
        let len = self.codons.len();
        self.codons[len - 1] = self.codons[len - 1].end_replace(
            self.codon_end,
            &dna.extract_padded_subsequence(0, self.codon_end as i64),
        );

        if self.codon_end >= dna.len() {
            self.codon_end = self.codon_end - dna.len();
            return;
        }

        self.codons.extend(
            DegenerateCodonSequence::from_dna(
                &dna.extract_subsequence(self.codon_end, dna.len()),
                0,
            )
            .codons,
        );

        // self.codon_end + dna.len())
        self.codon_end = mod_euclid(self.codon_end as i64 - dna.len() as i64, 3) as usize;
    }

    /// Add a dna sequence before an DegenerateCodonSequence sequence
    pub fn append_to_dna(&mut self, dna: &Dna) {
        // need to complete the first codon
        let len = self.codons.len();
        self.codons[len - 1] = self.codons[len - 1].start_replace(
            self.codon_start,
            &dna.extract_padded_subsequence(
                dna.len() as i64 - self.codon_start as i64,
                dna.len() as i64,
            ),
        );

        let start = mod_euclid(self.codon_start as i64 - dna.len() as i64, 3) as usize;
        if self.codon_start > dna.len() {
            self.codon_start = start;
            return;
        }

        let mut codons = DegenerateCodonSequence::from_dna(
            &dna.extract_subsequence(0, dna.len() - self.codon_start),
            start,
        )
        .codons;

        codons.extend(self.codons.clone());
        self.codons = codons;
        self.codon_start = start;
    }
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

// static AMINO_TO_DNA_LOSSY: phf::Map<u8, &'static str> = phf_map! {
//     b'A' => "GCN",
//     b'C' => "TGY",
//     b'D' => "GAY",
//     b'E' => "GAR",
//     b'F' => "TTY",
//     b'G' => "GGN",
//     b'H' => "CAY",
//     b'I' => "ATH",
//     b'L' => "YTN", // lossy
//     b'K' => "AAR",
//     b'M' => "ATG",
//     b'N' => "AAY",
//     b'P' => "CCN",
//     b'Q' => "CAR",
//     b'R' => "MGN", // lossy
//     b'S' => "WSN", // lossy
//     b'T' => "ACN",
//     b'V' => "GTN",
//     b'W' => "TGG",
//     b'Y' => "TAY",
//     b'*' => "TRR", // lossy
// };

// The standard ACGT nucleotides
// R: A/G, Y: T/C, S: C/G, W:A/T, M: C/A, K: G/T, B:T/C/G, D:A/G/T, H:A/C/T, V:A/C/G,
pub const NUCLEOTIDES: [u8; 15] = [
    b'A', b'C', b'G', b'T', b'N', b'R', b'Y', b'S', b'W', b'K', b'M', b'B', b'D', b'H', b'V',
];

pub const AMINOACIDS: [u8; 21] = [
    b'A', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'L', b'K', b'M', b'N', b'P', b'Q', b'R', b'S',
    b'T', b'V', b'W', b'Y', b'*',
];

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
        table[0b00000001 as usize] = b'A';
        table[0b00000010 as usize] = b'C';
        table[0b00000100 as usize] = b'G';
        table[0b00001000 as usize] = b'T';
        table[0b00001111 as usize] = b'N';
        table[0b00000101 as usize] = b'R';
        table[0b00001010 as usize] = b'Y';
        table[0b00000110 as usize] = b'S';
        table[0b00001001 as usize] = b'W';
        table[0b00001100 as usize] = b'K';
        table[0b00000011 as usize] = b'M';
        table[0b00001110 as usize] = b'B';
        table[0b00001101 as usize] = b'D';
        table[0b00001011 as usize] = b'H';
        table[0b00000111 as usize] = b'V';
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

pub fn compatible_nucleotides(x: u8, y: u8) -> bool {
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
    (MASK_TABLE[x as usize] & MASK_TABLE[y as usize]) != 0
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

// /// Dna that contains a degenerate codon (i.e. N, R, W ...)
// #[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
// #[derive(Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
// struct DegenerateCodonSequence {
//     // while Dna cannot be constructed with degenerate codons, we bypass it here
//     // so we can directly use most methods of Dna.
//     pub inner: Dna,
// }

// impl fmt::Debug for DegenerateCodonSequence {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(
//             f,
//             "DegenerateCodonSequence [{}]",
//             String::from_utf8_lossy(&self.inner.seq)
//         )
//     }
// }

// impl fmt::Display for DegenerateCodonSequence {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "{}", String::from_utf8_lossy(&self.inner.seq))
//     }
// }

// impl DegenerateCodonSequence {
//     fn __repr__(&self) -> String {
//         self.inner.__repr__()
//     }

//     pub fn get_string(&self) -> String {
//         self.inner.get_string()
//     }

//     pub fn translate(&self) -> Result<AminoAcid> {
//         self.inner.translate()
//     }

//     pub fn len(&self) -> usize {
//         self.inner.len()
//     }

//     pub fn is_empty(&self) -> bool {
//         self.inner.is_empty()
//     }

//     pub fn extend(&mut self, dna: &Dna) {
//         self.inner.extend(dna)
//     }

//     pub fn reverse(&mut self) {
//         self.inner.reverse()
//     }
// }

// impl DegenerateCodonSequence {
//     pub fn hamming_distance(&self, d: &DegenerateCodonSequence) -> usize {
//         self.inner
//             .seq
//             .iter()
//             .zip(&d.seq)
//             .map(|(&x, &y)| if compatible_nucleotides(x, y) { 0 } else { 1 })
//             .sum()
//     }

//     pub fn count_differences(&self, template: &DegenerateCodonSequence) -> usize {
//         self.hamming_distance(template)
//     }

//     // also works with slices
//     pub fn hamming_distance_u8slice(&self, d: &[u8]) -> usize {
//         self.seq
//             .iter()
//             .zip(d)
//             .map(|(&x, &y)| if compatible_nucleotides(x, y) { 0 } else { 1 })
//             .sum()
//     }

//     // also work with index of usize (A,C,G,T => 0,1,2,3)
//     pub fn hamming_distance_index_slice(&self, d: &[usize]) -> usize {
//         self.seq
//             .iter()
//             .zip(d)
//             .map(|(&x, &y)| {
//                 if compatible_nucleotides(x, NUCLEOTIDES[y]) {
//                     0
//                 } else {
//                     1
//                 }
//             })
//             .sum()
//     }

//     pub fn reverse_complement(&self) -> DegenerateCodonSequence {
//         DegenerateCodonSequence {
//             inner: self.inner.reverse_complement,
//         }
//     }

//     pub fn extract_subsequence(&self, start: usize, end: usize) -> DegenerateCodonSequence {
//         // Return dna[start:end]
//         DegenerateCodonSequence {
//             inner: self.inner.extract_subsequence,
//         }
//     }

//     pub fn extract_padded_subsequence(&self, start: i64, end: i64) -> DegenerateCodonSequence {
//         DegenerateCodonSequence {
//             inner: self.inner.extract_padded_subsequence(start, end),
//         }
//     }

//     pub fn v_alignment(
//         v: &Dna,
//         seq: &DegenerateCodonSequence,
//         align_params: &AlignmentParameters,
//     ) -> Option<Alignment> {
//         v_alignment(v, seq.inner, align_params)
//     }
// }

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
        write!(f, "Amino-Acid [{}]", String::from_utf8_lossy(&self.seq))
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

    // for a sequence of length 2, return the matrix idx
    // i.e 4*nucleotide[0] + nucleotide[1]
    pub fn to_matrix_idx(&self) -> Vec<usize> {
        vec![(4 * self.seq[0] + self.seq[1]).into()]
    }

    /// return all possible extremities for a given sequence
    /// the extremities are encoded in the index form for the 16x16 likelihood matrix
    /// hence (4*x[0] +  x[1], 4*x[x.len()] + x[x.len()+1])
    pub fn extremities(&self) -> Vec<(usize, usize)> {
        (0..16)
            .map(|x| ((4 * self.seq[0] + self.seq[1]).into(), x))
            .collect()
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
        debug_assert!(start + end <= 2);
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

    // pub fn position_differences_dna(sequence: &Dna, template: &Dna) -> Vec<usize> {
    //     // Return the position of the differences between sequence
    //     // and template
    //     template
    //         .seq
    //         .iter()
    //         .zip(sequence.seq.iter())
    //         .enumerate()
    //         .filter_map(
    //             |(index, (&v1, &v2))| {
    //                 if v1 != v2 {
    //                     Some(index)
    //                 } else {
    //                     None
    //                 }
    //             },
    //         )
    //         .collect()
    // }
}

impl AminoAcid {
    pub fn from_string(s: &str) -> Result<AminoAcid> {
        for &byte in s.as_bytes() {
            if !AMINOACIDS.contains(&byte) {
                // Handle the error if the byte is not in the map
                return Err(anyhow!(format!("Invalid byte: {}", byte)));
            }
        }

        return Ok(AminoAcid {
            seq: s.as_bytes().to_vec(),
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
        String::from_utf8_lossy(&self.seq).to_string()
    }

    #[staticmethod]
    #[pyo3(name = "from_string")]
    pub fn py_from_string(s: &str) -> Result<AminoAcid> {
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
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl DnaLike {
    pub fn to_dna(&self) -> Dna {
        self.inner.to_dna()
    }
    pub fn get_string(&self) -> String {
        self.inner.get_string()
    }

    fn __repr__(&self) -> String {
        self.get_string()
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
}

impl DnaLike {
    pub fn from_dna(seq: Dna) -> DnaLike {
        DnaLike {
            inner: DnaLikeEnum::from_dna(seq),
        }
    }

    pub fn from_amino_acid(seq: AminoAcid) -> DnaLike {
        DnaLike {
            inner: DnaLikeEnum::from_amino_acid(seq),
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

    pub fn reverse(&mut self) {
        self.inner.reverse();
    }

    pub fn extend(&mut self, other: DnaLike) {
        self.inner.extend(&other.inner).into()
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
