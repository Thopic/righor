//! This file is defining the functions to deal with reverse translation
//! and matrix making for the amino-acide sequence

use super::DNAMarkovChain;
use crate::shared::likelihood::{Matrix16, Matrix16x4, Matrix4, Matrix4x16};
use crate::shared::sequence::{
    codon_to_amino_acid, degenerate_dna_to_vec, degenerate_nucleotide, nucleotides_inv, AminoAcid,
    Dna, BLANK, BLANKN, NUCLEOTIDES,
};
use crate::shared::utils::mod_euclid;
use anyhow::Result;
use itertools::{iproduct, Itertools};
use serde::{Deserialize, Serialize};

use crate::shared::sequence::compatible_nucleotides;

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

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
/// Define a degenerated codon, ie a  vec of triplets of nucleotides
/// nucleotides are stored as 0,1,2,3 (A,C,G,T).
pub struct DegenerateCodon {
    pub triplets: Vec<[usize; 3]>,
}

impl DegenerateCodon {
    pub fn start_codon_matrix(
        &self,
        mc: &DNAMarkovChain,
        start: usize,            // the start of the sequence in the codon (0..2)
        first_nucleotide: usize, // the nucleotide before the codon start
    ) -> Matrix16x4 {
        let mut m = Matrix16x4::zeros();
        for cod in &self.triplets {
            if start == 0 {
                // everything possible
                for kk in 0..16 {
                    m[(kk, cod[2])] += mc.transition_matrix[(first_nucleotide, cod[0])]
                        * mc.transition_matrix[(cod[0], cod[1])]
                        * mc.transition_matrix[(cod[1], cod[2])];
                }
            }
            if start == 1 {
                for kk in 0..4 {
                    m[(4 * kk + cod[0], cod[2])] += mc.transition_matrix
                        [(first_nucleotide, cod[1])]
                        * mc.transition_matrix[(cod[1], cod[2])];
                }
            }
            if start == 2 {
                m[(4 * cod[0] + cod[1], cod[2])] +=
                    mc.transition_matrix[(first_nucleotide, cod[2])];
            }
        }
        m
    }

    pub fn middle_codon_matrix(&self, mc: &DNAMarkovChain) -> Matrix4 {
        let mut m = Matrix4::zeros();
        for cod in &self.triplets {
            for ll in 0..4 {
                m[(ll, cod[2])] += mc.transition_matrix[(ll, cod[0])]
                    * mc.transition_matrix[(cod[0], cod[1])]
                    * mc.transition_matrix[(cod[1], cod[2])];
            }
        }
        m
    }

    pub fn end_codon_matrix(
        &self,
        mc: &DNAMarkovChain,
        end: usize, // the end of the sequence in the codon (0: full cod., 2: only 1 nt)
    ) -> Matrix4x16 {
        let mut m = Matrix4x16::zeros();
        for cod in &self.extract_subsequence(0, 3 - end).triplets {
            if end == 0 {
                for ii in 0..4 {
                    m[(ii, 4 * cod[1] + cod[2])] += mc.transition_matrix[(ii, cod[0])]
                        * mc.transition_matrix[(cod[0], cod[1])]
                        * mc.transition_matrix[(cod[1], cod[2])];
                }
            }
            if end == 1 {
                for ii in 0..4 {
                    m[(ii, 4 * cod[0] + cod[1])] +=
                        mc.transition_matrix[(ii, cod[0])] * mc.transition_matrix[(cod[0], cod[1])];
                }
            }
            if end == 2 {
                for ii in 0..4 {
                    // this "trick" allow us to not have to care about the
                    // second-to-last codon.
                    m[(ii, 4 * ii + cod[0])] += mc.transition_matrix[(ii, cod[0])];
                }
            }
        }
        m
    }

    ///  if there's only one amino-acid (or less) in the sequence
    pub fn lonely_codon_matrix(
        &self,
        mc: &DNAMarkovChain,
        start: usize,
        end: usize,
        first_nucleotide: usize,
    ) -> Matrix16 {
        let mut m = Matrix16::zeros();

        // the constraint on start & end are not the same,
        // start is before the sequence, so we need to know the first
        // nucleotides of the codon, the end index is in the sequence
        // so we shouldn't count anything that comes after
        for cod in &self.extract_subsequence(0, 3 - end).triplets {
            match (start, end) {
                (0, 0) => {
                    // full codon
                    for kk in 0..16 {
                        m[(kk, 4 * cod[1] + cod[2])] += mc.transition_matrix
                            [(first_nucleotide, cod[0])]
                            * mc.transition_matrix[(cod[0], cod[1])]
                            * mc.transition_matrix[(cod[1], cod[2])];
                    }
                }
                (1, 0) => {
                    // last 2 nucleotides
                    for kk in 0..16 {
                        m[(kk, 4 * cod[1] + cod[2])] += mc.transition_matrix
                            [(first_nucleotide, cod[1])]
                            * mc.transition_matrix[(cod[1], cod[2])];
                    }
                }
                (0, 1) => {
                    for kk in 0..16 {
                        m[(kk, 4 * cod[0] + cod[1])] += mc.transition_matrix
                            [(first_nucleotide, cod[0])]
                            * mc.transition_matrix[(cod[0], cod[1])];
                    }
                }
                (2, 0) => {
                    for kk in 0..16 {
                        m[(kk, 4 * (kk % 4) + cod[2])] +=
                            mc.transition_matrix[(first_nucleotide, cod[2])];
                    }
                }
                (0, 2) => {
                    for kk in 0..16 {
                        m[(kk, 4 * (kk % 4) + cod[0])] +=
                            mc.transition_matrix[(first_nucleotide, cod[0])];
                    }
                }
                (1, 1) => {
                    for kk in 0..16 {
                        m[(kk, 4 * (kk % 4) + cod[1])] +=
                            mc.transition_matrix[(first_nucleotide, cod[1])];
                    }
                }
                // (1, 2) | (2, 1) => {
                //     Matrix16::identity();
                // }
                _ => {
                    panic!("Probably shouldn't happen");
                }
            }
        }
        m
    }

    /// For the reversed case we need to be a bit more careful,
    /// Instead of returning the matrix (-2, len-2), I will return
    /// the matrix (len - 2, -2), then transpose after mult.
    pub fn reversed_start_codon_matrix(
        &self,
        mc: &DNAMarkovChain,
        end: usize,
        first_nucleotide: usize,
    ) -> Matrix16x4 {
        let mut m = Matrix16x4::zeros();
        for cod in &self.extract_subsequence(0, 3 - end).triplets {
            if end == 0 {
                for next in 0..4 {
                    m[(4 * cod[1] + cod[2], next)] += mc.transition_matrix
                        [(first_nucleotide, cod[2])]
                        * mc.transition_matrix[(cod[2], cod[1])]
                        * mc.transition_matrix[(cod[1], cod[0])]
                        * mc.transition_matrix[(cod[0], next)];
                }
            }
            if end == 1 {
                for next in 0..4 {
                    m[(4 * cod[0] + cod[1], next)] += mc.transition_matrix
                        [(first_nucleotide, cod[1])]
                        * mc.transition_matrix[(cod[1], cod[0])]
                        * mc.transition_matrix[(cod[0], next)];
                }
            }
            if end == 2 {
                for next in 0..4 {
                    m[(4 * next + cod[0], next)] += mc.transition_matrix
                        [(first_nucleotide, cod[0])]
                        * mc.transition_matrix[(cod[0], next)];
                }
            }
        }
        m
    }

    // pub fn reversed_second_codon_matrix(&self, mc: &DNAMarkovChain) -> Matrix16x4 {
    //     let mut m = Matrix16x4::zeros();
    //     for cod in self.triplets.iter() {
    //         for cod3 in 0..4 {
    //             // the first nucleotide of the "next codon"
    //             m[(4 * cod[2] + cod3, cod[0])] += mc.transition_matrix[(cod3, cod[2])]
    //                 * mc.transition_matrix[(cod[2], cod[1])]
    //                 * mc.transition_matrix[(cod[1], cod[0])];
    //         }
    //     }
    //     m
    // }

    pub fn reversed_middle_codon_matrix(&self, mc: &DNAMarkovChain) -> Matrix4 {
        let mut m = Matrix4::zeros();
        for cod in &self.triplets {
            for next in 0..4 {
                // the first nucleotide of the "next codon"
                m[(cod[2], next)] += mc.transition_matrix[(cod[2], cod[1])]
                    * mc.transition_matrix[(cod[1], cod[0])]
                    * mc.transition_matrix[(cod[0], next)];
            }
        }
        m
    }

    pub fn reversed_end_codon_matrix(&self, mc: &DNAMarkovChain, start: usize) -> Matrix4x16 {
        let mut m = Matrix4x16::zeros();
        for cod in &self.triplets {
            match start {
                0 => {
                    // full codon
                    for ll in 0..16 {
                        m[(cod[2], ll)] += mc.transition_matrix[(cod[2], cod[1])]
                            * mc.transition_matrix[(cod[1], cod[0])];
                    }
                }
                1 => {
                    for ll in 0..4 {
                        m[(cod[2], 4 * ll + cod[0])] += mc.transition_matrix[(cod[2], cod[1])];
                    }
                }
                2 => {
                    // for ll in 0..4 {
                    m[(cod[2], 4 * cod[0] + cod[1])] += 1.;
                    //                        }
                }
                _ => {
                    panic!("Invalid dna end (reversed_end_codon_matrix)")
                }
            }
        }
        m
    }

    pub fn reversed_lonely_codon_matrix(
        &self,
        mc: &DNAMarkovChain,
        start: usize,
        end: usize,
        first_nucleotide: usize,
    ) -> Matrix16 {
        let mut m = Matrix16::zeros();

        for cod in &self.extract_subsequence(0, 3 - end).triplets {
            match (start, end) {
                (0, 0) => {
                    // full codon
                    for kk in 0..16 {
                        m[(kk, 4 * cod[1] + cod[2])] += mc.transition_matrix
                            [(first_nucleotide, cod[2])]
                            * mc.transition_matrix[(cod[2], cod[1])]
                            * mc.transition_matrix[(cod[1], cod[0])];
                    }
                }
                (1, 0) => {
                    // last 2 nucleotides
                    for kk in 0..4 {
                        m[(4 * kk + cod[0], 4 * cod[1] + cod[2])] += mc.transition_matrix
                            [(first_nucleotide, cod[2])]
                            * mc.transition_matrix[(cod[2], cod[1])];
                    }
                }
                (0, 1) => {
                    for kk in 0..16 {
                        m[(kk, 4 * cod[0] + cod[1])] += mc.transition_matrix
                            [(first_nucleotide, cod[1])]
                            * mc.transition_matrix[(cod[1], cod[0])];
                    }
                }
                (2, 0) => {
                    m[(4 * cod[0] + cod[1], 4 * cod[1] + cod[2])] +=
                        mc.transition_matrix[(first_nucleotide, cod[2])];
                }
                (0, 2) => {
                    for kk in 0..16 {
                        m[(kk, 4 * (kk % 4) + cod[0])] +=
                            mc.transition_matrix[(first_nucleotide, cod[0])];
                    }
                }
                (1, 1) => {
                    for kk in 0..4 {
                        m[(4 * kk + cod[0], 4 * cod[0] + cod[1])] +=
                            mc.transition_matrix[(first_nucleotide, cod[1])];
                    }
                }
                _ => {
                    panic!("Probably shouldn't happen");
                }
            }
        }
        m
    }

    /// last weird case, the second codon if the reversed sequence
    /// is only 2 codon long
    pub fn reversed_second_end_codon_matrix(&self, mc: &DNAMarkovChain, end: usize) -> Matrix16 {
        let mut m = Matrix16::zeros();
        for cod in &self.triplets {
            match end {
                0 => {
                    for cod3 in 0..4 {
                        for ll in 0..16 {
                            m[(4 * cod3 + cod[2], ll)] += mc.transition_matrix[(cod3, cod[2])]
                                * mc.transition_matrix[(cod[2], cod[1])]
                                * mc.transition_matrix[(cod[1], cod[0])];
                        }
                    }
                }
                1 => {
                    for cod3 in 0..4 {
                        for ll in 0..4 {
                            // the first nucleotide of the "next codon"
                            m[(4 * cod[2] + cod3, 4 * ll + cod[0])] += mc.transition_matrix
                                [(cod3, cod[2])]
                                * mc.transition_matrix[(cod[2], cod[1])];
                        }
                    }
                }
                2 => {
                    for cod3 in 0..4 {
                        m[(4 * cod[2] + cod3, 4 * cod[0] + cod[1])] +=
                            mc.transition_matrix[(cod3, cod[2])];
                    }
                }
                _ => {
                    panic!("end > 2, this shouldn't happen.")
                }
            }
        }
        m
    }

    /// Remove the start/end of a codon (replace by BLANK) and uniquify
    pub fn extract_subsequence(&self, start: usize, end: usize) -> DegenerateCodon {
        debug_assert!(end <= 3);
        let mirror_end = 3 - end;
        match (start, mirror_end) {
            (0, 0) => DegenerateCodon {
                triplets: self.triplets.iter().copied().unique().collect(),
            },
            (0, 1) => DegenerateCodon {
                triplets: self
                    .triplets
                    .iter()
                    .copied()
                    .map(|x| [x[0], x[1], BLANK])
                    .unique()
                    .collect(),
            },
            (0, 2) => DegenerateCodon {
                triplets: self
                    .triplets
                    .iter()
                    .copied()
                    .map(|x| [x[0], BLANK, BLANK])
                    .unique()
                    .collect(),
            },
            (1, 0) => DegenerateCodon {
                triplets: self
                    .triplets
                    .iter()
                    .copied()
                    .map(|x| [BLANK, x[1], x[2]])
                    .unique()
                    .collect(),
            },
            (1, 1) => DegenerateCodon {
                triplets: self
                    .triplets
                    .iter()
                    .copied()
                    .map(|x| [BLANK, x[1], BLANK])
                    .unique()
                    .collect(),
            },
            (2, 0) => DegenerateCodon {
                triplets: self
                    .triplets
                    .iter()
                    .copied()
                    .map(|x| [BLANK, BLANK, x[2]])
                    .unique()
                    .collect(),
            },
            (2, 1) | (1, 2) | (3, 0) | (0, 3) => DegenerateCodon {
                triplets: vec![[BLANK, BLANK, BLANK]],
            },
            _ => {
                panic!("Degenerate Codon extract_subsequence received invalid start/end");
            }
        }
    }

    pub fn translate(&self) -> u8 {
        let mut b = None;
        for v in &self.triplets {
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

    /// Return all the possible codons (64), in a fixed order
    pub fn all_codons() -> Vec<[usize; 3]> {
        let mut v = vec![];
        for n1 in 0..4 {
            for n2 in 0..4 {
                for n3 in 0..4 {
                    v.push([n1, n2, n3]);
                }
            }
        }
        v
    }

    /// Return the index of a given nucleotide triplet. `all_codons()[get_index(x)] = x`
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
                a if (128..192).contains(&a) => vec![[
                    ((a - 128) % 4) as usize,
                    (((a - 128) / 4) % 4) as usize,
                    ((a - 128) / 16) as usize,
                ]],
                _ => panic!("Wrong amino-acid sequence {x:?}"),
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
            triplets: iproduct!(
                degenerate_dna_to_vec(x[0]),
                degenerate_dna_to_vec(x[1]),
                degenerate_dna_to_vec(x[2])
            )
            .map(|(x, y, z)| [x, y, z])
            .collect(),
        }
    }

    /// For every possible end to the degenerate codon (two nucl.)
    /// give a sub degeneratecodon that matches them.
    /// For ex [[0,1,2], [1,1,2], [2,2,2]]
    /// Will return vec![(4*1 + 2, [[0,1,2], [1,1,2]]), (4*2+2, [[2,2,2]])]
    pub fn fix_end_two(&self) -> Vec<(usize, DegenerateCodon)> {
        let unique_ends = self.triplets.iter().map(|x| 4 * x[1] + x[2]).unique();
        let mut results = vec![];
        for end in unique_ends {
            let cod = DegenerateCodon {
                triplets: self
                    .triplets
                    .iter()
                    .copied()
                    .filter(|x| 4 * x[1] + x[2] == end)
                    .collect(),
            };
            results.push((end, cod));
        }
        results
    }

    /// For every possible end to the degenerate codon (one nucl.)
    /// give the sub degeneratecodon that matches them.
    /// For ex [[0,1,2], [1,1,2], [2,2,1]]
    /// Will return vec![(2, [[0,1,2], [1,1,2]]), (1, [[2,2,1]])]
    pub fn fix_end_one(&self) -> Vec<(usize, DegenerateCodon)> {
        let unique_ends = self.triplets.iter().map(|x| x[2]).unique();
        let mut results = vec![];
        for end in unique_ends {
            let cod = DegenerateCodon {
                triplets: self
                    .triplets
                    .iter()
                    .copied()
                    .filter(|x| x[2] == end)
                    .collect(),
            };
            results.push((end, cod));
        }
        results
    }

    /// For every possible two-nuc starts of the degenerate codon
    /// give the sub codons that matches them
    /// For ex [[0, 1, 2], [1,1,2], [1,1,1]]
    /// Will return vec![(4*0 + 1, [0, 1, 2]), (4*1+1, [[1,1,2], [1,1,2]])]
    pub fn fix_start_two(&self) -> Vec<(usize, DegenerateCodon)> {
        let unique_starts = self.triplets.iter().map(|x| 4 * x[0] + x[1]).unique();
        let mut results = vec![];
        for start in unique_starts {
            let cod = DegenerateCodon {
                triplets: self
                    .triplets
                    .iter()
                    .copied()
                    .filter(|x| 4 * x[0] + x[1] == start)
                    .collect(),
            };
            results.push((start, cod));
        }
        results
    }

    pub fn fix_start_one(&self) -> Vec<(usize, DegenerateCodon)> {
        let unique_starts = self.triplets.iter().map(|x| x[0]).unique();
        let mut results = vec![];
        for start in unique_starts {
            let cod = DegenerateCodon {
                triplets: self
                    .triplets
                    .iter()
                    .copied()
                    .filter(|x| x[0] == start)
                    .collect(),
            };
            results.push((start, cod));
        }
        results
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
    pub fn hamming_distance(&self, seq: &[u8], start: usize, end: usize) -> usize {
        debug_assert!(seq.len() == 3 - start - end);

        self.triplets
            .iter()
            .map(|x| {
                seq.iter()
                    .zip(&x[start..3 - end])
                    .map(|(&u, &v)| {
                        if compatible_nucleotides(u, NUCLEOTIDES[v]) {
                            0
                        } else {
                            1
                        }
                    })
                    .sum()
            })
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
    /// return all possible extremities for a given sequence
    /// the extremities are encoded in the index form for the 16x16 likelihood matrix
    /// hence `(4*x[0] +  x[1], 4*x[x.len()] + x[x.len()+1])`
    pub fn valid_extremities(&self) -> Vec<(usize, usize)> {
        let mut v = vec![];
        for idx_left in 0..16 {
            let mut seq = self.clone();
            seq.append_to_dna(&Dna::from_matrix_idx(idx_left));
            let idx_right_vec = seq
                .extract_subsequence(seq.len() - 2, seq.len())
                .to_matrix_idx();
            for idx_right in idx_right_vec {
                v.push((idx_left, idx_right));
            }
        }
        v
    }

    pub fn pad_right(&mut self, n: usize) {
        // X for undefined amino-acid
        self.extend_dna(&Dna {
            seq: vec![BLANKN; n],
        });
    }

    pub fn pad_left(&mut self, n: usize) {
        self.append_to_dna(&Dna {
            seq: vec![BLANKN; n],
        });
    }

    /// Make an amino-acid sequence into an `UndefinedDna` sequence
    pub fn from_aminoacid(aa: &AminoAcid) -> DegenerateCodonSequence {
        DegenerateCodonSequence {
            codons: aa
                .seq
                .iter()
                .map(|&x| DegenerateCodon::from_amino(x))
                .collect(),
            codon_start: aa.start,
            codon_end: aa.end,
        }
    }

    /// Same but from a &u8
    pub fn from_aminoacid_u8(aa: &[u8], start: usize, end: usize) -> DegenerateCodonSequence {
        DegenerateCodonSequence {
            codons: aa.iter().map(|&x| DegenerateCodon::from_amino(x)).collect(),
            codon_start: start,
            codon_end: end,
        }
    }

    /// Make a (non-degenerate) nucleotide sequence into a
    /// `DegenerateCodonSequence` sequence.
    /// For example ATCG, start = 1 would give
    /// [[AAT, CAT, GAT, TAT], [CGA, CGC, CGT, CGN]]
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
                .map(DegenerateCodon::from_u8)
                .collect(),
            codon_start: start,
            codon_end: mod_euclid(3 - seq.len() as i64 - start as i64, 3) as usize,
        }
    }

    pub fn translate(&self) -> Result<AminoAcid> {
        Ok(AminoAcid {
            seq: self.codons.iter().map(DegenerateCodon::translate).collect(),
            start: self.codon_start,
            end: self.codon_end,
        })
    }

    pub fn to_matrix_idx(&self) -> Vec<usize> {
        debug_assert!(self.len() == 2);

        let mut result = vec![];

        if self.codon_start == 0 {
            for cod in &self.codons[0].triplets {
                result.push(4 * cod[0] + cod[1]);
            }
        } else if self.codon_start == 1 {
            for cod in &self.codons[0].triplets {
                result.push(4 * cod[1] + cod[2]);
            }
        } else if self.codon_start == 2 {
            for cod1 in &self.codons[0].triplets {
                for cod2 in &self.codons[1].triplets {
                    result.push(4 * cod1[2] + cod2[0]);
                }
            }
        }
        result
    }

    /// lossy process, remove some information about the codon
    pub fn to_dna(&self) -> Dna {
        let sequence = Dna {
            seq: self.codons.iter().flat_map(|aa| aa.to_dna().seq).collect(),
        };
        //        println!("{:?}", sequence);
        sequence.extract_subsequence(self.codon_start, self.len() + self.codon_start)
    }

    /// return all dna sequences possible (can be huge)
    pub fn to_dnas(&self) -> Vec<Dna> {
        let mut all_nts = vec![Dna::new()];
        for (cod_nb, codons) in self.codons.iter().enumerate() {
            let mut new_combinations = Vec::new();
            let mut new_codons = codons.clone();
            if cod_nb == 0 && cod_nb == self.codons.len() - 1 {
                new_codons = new_codons.extract_subsequence(self.codon_start, 3 - self.codon_end);
            } else if cod_nb == self.codons.len() - 1 {
                new_codons = new_codons.extract_subsequence(0, 3 - self.codon_end);
            } else if cod_nb == 0 {
                new_codons = new_codons.extract_subsequence(self.codon_start, 3);
            }
            for cod in &new_codons.triplets {
                for nt in &all_nts {
                    let mut new_seq = nt.seq.clone();
                    new_seq.extend(cod.iter().filter(|&x| *x != BLANK).map(|&x| NUCLEOTIDES[x]));
                    new_combinations.push(Dna { seq: new_seq });
                }
            }

            all_nts = new_combinations;
        }
        all_nts
    }

    /// Extract a subsequence from the dna.
    /// [AATNAT][4:6] ->  [.AT] (with `start_codon = 1`)
    pub fn strict_extract_subsequence(&self, start: usize, end: usize) -> DegenerateCodonSequence {
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
        // We also clean up / simplify the first and last codon.
        // For example [NAT][1:3] is going to give [.AT] rather than [AAT,CAT,GAT,TAT]
        debug_assert!(end <= self.len());

        let shift_start = start + self.codon_start;
        let shift_end = end + self.codon_start;

        let aa_start = shift_start / 3;
        // we check where is the last element, divide by 3 then add 1.
        let aa_end = (shift_end + 3 - 1) / 3;

        let mut new_codons = self.codons[aa_start..aa_end].to_vec();
        if let Some(first) = new_codons.first_mut() {
            *first = self.codons[aa_start].extract_subsequence(shift_start % 3, 3);
        }

        if let Some(last) = new_codons.last_mut() {
            *last = self.codons[aa_end - 1].extract_subsequence(0, 3 - (3 * (aa_end) - shift_end));
        }
        DegenerateCodonSequence {
            codons: new_codons,
            codon_start: shift_start % 3,
            codon_end: 3 * (aa_end) - shift_end,
        }
    }

    /// Extract a subsequence from the dna. Keep the codons
    /// [AATNAT][4:6] ->  [AAT,CAT,GAT,TAT] (with `start_codon = 1`)
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
        // We also clean up / simplify the first and last codon.
        // For example [NAT][1:3] is going to give [.AT] rather than [AAT,CAT,GAT,TAT]
        debug_assert!(end <= self.len());

        let shift_start = start + self.codon_start;
        let shift_end = end + self.codon_start;

        let aa_start = shift_start / 3;
        // we check where is the last element, divide by 3 then add 1.
        let aa_end = (shift_end + 3 - 1) / 3;

        let new_codons = self.codons[aa_start..aa_end].to_vec();
        DegenerateCodonSequence {
            codons: new_codons,
            codon_start: shift_start % 3,
            codon_end: 3 * (aa_end) - shift_end,
        }
    }

    /// Return dna[start:end] but padded with N if `start < 0` or `end >= dna.len()`
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

    /// Count the number of difference between a slice of the sequence
    /// and a slice of the template
    /// Hamming distance, choosing the most favorable codon each time
    pub fn count_differences_slice(&self, d: &Dna, start_d: usize, end_d: usize) -> usize {
        debug_assert!(end_d - start_d == self.codons.len() - self.codon_end - self.codon_start);
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
                &d.seq[current + start_d..current + start_d + 3 - start - end],
                start,
                end,
            );
            current += 3 - start - end;
        }
        distance
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
                &template.seq[current..current + 3 - start - end],
                start,
                end,
            );
            current += 3 - start - end;
        }
        distance
    }

    /// Return all possiblies extremities of the sequence (effectively
    /// fixing the first and last codon)
    ///          sssssssssss
    ///        XX         XX
    /// return   SSSSSSSSSSS
    pub fn fix_extremities(&self) -> Vec<(usize, usize, DegenerateCodonSequence)> {
        self.left_extremities() // need to start with left to increase size first
            .into_iter()
            .flat_map(|(xleft, seq_left)| {
                seq_left
                    .right_extremities()
                    .into_iter()
                    .map(move |(xright, seq)| {
                        // here strict_extract_subsequence, we want the count
                        // to be right
                        (xleft, xright, seq.strict_extract_subsequence(2, seq.len()))
                    })
            })
            .collect()
    }

    /// Return all possible left extremities of the sequence
    ///   sssssssss
    /// XX
    /// `return [(XX, XXsssssssss) for all XX]`
    pub fn left_extremities(&self) -> Vec<(usize, DegenerateCodonSequence)> {
        let mut results = vec![];
        if self.is_empty() {
            // no codons
            for idx_left in 0..16 {
                let mut s = self.clone();
                let new_start_codon = DegenerateCodon {
                    triplets: vec![[BLANK, idx_left / 4, idx_left % 4]],
                };
                s.codons.insert(0, new_start_codon);
                s.codon_start = 1;
                results.push((idx_left, s));
            }
            return results;
        }

        let first_codon = self.codons[0].clone();

        if self.codon_start == 2 {
            for (idx_left, cod) in first_codon.fix_start_two() {
                let mut s = self.clone();
                s.codon_start = 0;
                s.codons[0] = cod;
                results.push((idx_left, s));
            }
        } else if self.codon_start == 1 {
            for nuc1 in 0..4 {
                for (nuc2, cod) in first_codon.fix_start_one() {
                    let mut s = self.clone();
                    s.codons[0] = cod;
                    let new_start_codon = DegenerateCodon {
                        triplets: vec![[BLANK, BLANK, nuc1]],
                    };
                    s.codons.insert(0, new_start_codon);
                    s.codon_start = 2;
                    results.push((4 * nuc1 + nuc2, s));
                }
            }
        } else if self.codon_start == 0 {
            for idx_left in 0..16 {
                let mut s = self.clone();
                let new_start_codon = DegenerateCodon {
                    triplets: vec![[BLANK, idx_left / 4, idx_left % 4]],
                };
                s.codons.insert(0, new_start_codon);
                s.codon_start = 1;
                results.push((idx_left, s));
            }
        }
        results
    }

    /// Return all possible right extremities of the sequence
    /// sssssssss
    ///        XX
    /// Return [(XX, sssssssXX)]
    pub fn right_extremities(&self) -> Vec<(usize, DegenerateCodonSequence)> {
        debug_assert!(self.len() >= 2);

        let mut codons_minus_last = self.codons.clone();
        let last_codon = codons_minus_last.pop().unwrap();

        if self.codon_end == 0 {
            last_codon
                .fix_end_two()
                .into_iter()
                .map(|(idx, cod)| {
                    let mut new_codons = codons_minus_last.clone();
                    new_codons.push(cod);
                    (
                        idx,
                        DegenerateCodonSequence {
                            codons: new_codons,
                            codon_end: self.codon_end,
                            codon_start: self.codon_start,
                        },
                    )
                })
                .collect()
        } else if self.codon_end == 1 {
            // easy case
            last_codon
                .fix_start_two()
                .into_iter()
                .map(|(x, _)| {
                    let mut new_codons = codons_minus_last.clone();
                    new_codons.push(DegenerateCodon {
                        triplets: vec![[x / 4, x % 4, BLANK]],
                    });
                    (
                        x,
                        DegenerateCodonSequence {
                            codons: new_codons,
                            codon_end: self.codon_end,
                            codon_start: self.codon_start,
                        },
                    )
                })
                .collect()
        } else if self.codon_end == 2 {
            let mut results = vec![];
            let mut codons_minus_two = codons_minus_last.clone();
            // len >= 2 so it's  fine.
            let lastlast_codon = codons_minus_two.pop().unwrap();

            for (last_nuc, _) in last_codon.fix_start_one() {
                for (lastlast_nuc, cod) in lastlast_codon.fix_end_one() {
                    let end_idx = 4 * lastlast_nuc + last_nuc;
                    let mut new_codons = codons_minus_two.clone();
                    new_codons.push(cod);
                    new_codons.push(DegenerateCodon {
                        triplets: vec![[last_nuc, BLANK, BLANK]],
                    });
                    results.push((
                        end_idx,
                        DegenerateCodonSequence {
                            codons: new_codons,
                            codon_start: self.codon_start,
                            codon_end: self.codon_end,
                        },
                    ));
                }
            }
            return results;
        } else {
            panic!("Wrong value for codon_end");
        }
    }

    pub fn len(&self) -> usize {
        3 * self.codons.len() - self.codon_start - self.codon_end
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reverse(&mut self) {
        self.codons = self
            .codons
            .iter()
            .rev()
            .map(DegenerateCodon::reverse)
            .collect();
        std::mem::swap(&mut self.codon_end, &mut self.codon_start);
    }

    pub fn extend(&mut self, _dna: &DegenerateCodonSequence) {
        // this is very weird because frame shift, should happend
        unimplemented!("Appending two DegenerateCodonSequence shouldn't happen.")
    }

    pub fn extended_with_dna(&self, dna: &Dna) -> DegenerateCodonSequence {
        let mut s = self.clone();
        s.extend_dna(dna);
        s
    }

    /// Add a dna sequence at the end of an `DegenerateCodonSequence` sequence
    pub fn extend_dna(&mut self, dna: &Dna) {
        //               <---------->xxxxxx
        //             ..............xxxxxx
        //              x  x  x  x  x  x  x
        // Note: this is complicated to implement as we need to deal
        // with all the painful edge cases (empty dna, very short dna)
        // need extensive testing

        if self.is_empty() {
            *self = DegenerateCodonSequence::from_dna(dna, 0);
        }

        // if len(dna) < self.codon_end we can't fully complete
        let len = self.codons.len();
        self.codons[len - 1] = self.codons[len - 1].end_replace(
            self.codon_end,
            &dna.extract_padded_subsequence(0, self.codon_end as i64),
        );

        if self.codon_end >= dna.len() {
            self.codon_end -= dna.len();
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

    /// Add a dna sequence before a `DegenerateCodonSequence` sequence
    pub fn append_to_dna(&mut self, dna: &Dna) {
        // need to complete the first codon

        if self.is_empty() {
            *self = DegenerateCodonSequence::from_dna(dna, 0);
            return;
        }

        self.codons[0] = self.codons[0].start_replace(
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
