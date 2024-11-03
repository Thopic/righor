// Deal with the subtleties of the Markov chain on Dna or Amino-Acid structure

//use crate::shared::distributions::calc_steady_state_dist;
use crate::shared::likelihood::Likelihood;
use crate::shared::likelihood::{Matrix16, Matrix16x4, Matrix4, Matrix4x16, Vector4};
use crate::shared::sequence::{compatible_nucleotides, is_degenerate, AMINOACIDS, NUCLEOTIDES};
use crate::shared::sequence::{AminoAcid, Dna, DnaLike, DnaLikeEnum};
use crate::shared::{
    amino_acids::DegenerateCodon, amino_acids::DegenerateCodonSequence, nucleotides_inv,
    utils::normalize_transition_matrix,
};
use anyhow::Result;
use ndarray::Array2;
use serde::{de, ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

#[derive(Clone, PartialEq, Debug, Default)]
pub struct DNAMarkovChain {
    pub transition_matrix: Array2<f64>,
    // pre-computed data for dealing with the degenerate nucleotides
    pub degenerate_matrix: Vec<Matrix4>, // likelihood matrix for the degenerate dna
    aa_lone_rev: HashMap<(u8, usize, usize, usize), Matrix16>,
    aa_lone: HashMap<(u8, usize, usize, usize), Matrix16>,
    aa_start_rev: HashMap<(u8, usize, usize), Matrix16x4>,
    aa_start: HashMap<(u8, usize, usize), Matrix16x4>,
    aa_middle_rev: HashMap<u8, Matrix4>,
    aa_middle: HashMap<u8, Matrix4>,
    aa_end_rev: HashMap<(u8, usize), Matrix4x16>,
    aa_end: HashMap<(u8, usize), Matrix4x16>,
    pub end_degenerate_vector: Vec<Vector4>,

    pub reverse: bool,
}

impl Serialize for DNAMarkovChain {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("DNAMarkovChain", 1)?;
        state.serialize_field("transition_matrix", &self.transition_matrix)?;
        state.serialize_field("reverse", &self.reverse)?;
        state.end()
    }
}

// Implement custom Deserialize to initialize `computed_field`
impl<'de> Deserialize<'de> for DNAMarkovChain {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Temporarily deserialize only `field`, skipping `computed_field`
        #[derive(Deserialize)]
        struct MyStructData {
            pub transition_matrix: Array2<f64>,
            pub reverse: bool,
        }

        // Deserialize `field` and initialize `computed_field`
        let data = MyStructData::deserialize(deserializer)?;

        DNAMarkovChain::new(&data.transition_matrix, data.reverse).map_err(de::Error::custom)
    }
}

impl DNAMarkovChain {
    pub fn reinitialize(&mut self) -> Result<Self> {
        Self::new(&self.transition_matrix, self.reverse)
    }

    pub fn new(transition_matrix: &Array2<f64>, reverse: bool) -> Result<DNAMarkovChain> {
        let mut mc = DNAMarkovChain::default();
        mc.reverse = reverse;
        mc.precompute(transition_matrix)?;
        Ok(mc)
    }

    pub fn likelihood(&self, sequence: &DnaLike, first_nucleotide: usize) -> Likelihood {
        match sequence.clone().into() {
            DnaLikeEnum::Known(s) => self.likelihood_dna(&s, first_nucleotide),
            DnaLikeEnum::Ambiguous(s) => self.likelihood_degenerate(&s, first_nucleotide),
            DnaLikeEnum::Protein(s) => self.likelihood_aminoacid(&s, first_nucleotide),
        }
    }

    pub fn update(
        &self,
        sequence: &DnaLike,
        first_nucleotide: usize,
        likelihood: f64,
    ) -> Array2<f64> {
        match (*sequence).clone().into() {
            DnaLikeEnum::Known(s) => self.update_dna(&s, first_nucleotide, likelihood),
            DnaLikeEnum::Ambiguous(s) => self.update_degenerate(&s, first_nucleotide, likelihood),
            DnaLikeEnum::Protein(s) => self.update_aminoacid(&s, first_nucleotide, likelihood),
        }
    }

    pub fn precompute(&mut self, transition_matrix: &Array2<f64>) -> Result<()> {
        self.transition_matrix = normalize_transition_matrix(transition_matrix)?;
        //	let steady_state = calc_steady_state_dist(transition_matrix)?;
        //        self.initial_distribution = Vector4::new(steady_state[0], steady_state[1], steady_state[2], steady_state[3]);
        self.precompute_degenerate();
        self.precompute_amino_acid();
        Ok(())
    }
}

// functions specific to non-degenerate dna sequences
impl DNAMarkovChain {
    pub fn likelihood_dna(&self, s: &Dna, first: usize) -> Likelihood {
        if s.len() == 0 {
            return Likelihood::Scalar(1.);
        }

        let mut new_s = s.clone();
        if self.reverse {
            new_s.reverse()
        }

        let mut proba = self.transition_matrix[[first, nucleotides_inv(new_s.seq[0])]];
        for ii in 1..new_s.len() {
            proba *= self.transition_matrix[[
                nucleotides_inv(new_s.seq[ii - 1]),
                nucleotides_inv(new_s.seq[ii]),
            ]];
        }
        Likelihood::Scalar(proba)
    }

    pub fn update_dna(&self, s: &Dna, first: usize, likelihood: f64) -> Array2<f64> {
        let mut transition_mat = Array2::zeros((4, 4));
        let mut new_s = s.clone();
        if self.reverse {
            new_s.reverse()
        }
        transition_mat[[first, nucleotides_inv(new_s.seq[0])]] += likelihood;
        for ii in 1..new_s.len() {
            transition_mat[[
                nucleotides_inv(new_s.seq[ii - 1]),
                nucleotides_inv(new_s.seq[ii]),
            ]] += likelihood
        }
        transition_mat
    }
}

// functions specific to degenerate dna sequences
impl DNAMarkovChain {
    pub fn likelihood_degenerate(&self, s: &Dna, first: usize) -> Likelihood {
        if s.len() == 0 {
            return Likelihood::Scalar(1.);
        }

        let mut new_s = s.clone();
        if self.reverse {
            new_s.reverse()
        }

        let mut vector_proba = Vector4::new(0., 0., 0., 0.);
        vector_proba[first] = 1.;
        let mut vector_proba = vector_proba.transpose()
            * self.get_degenerate_matrix(first, nucleotides_inv(new_s.seq[0]));
        for ii in 1..new_s.len() {
            vector_proba = vector_proba
                * self.get_degenerate_matrix(
                    nucleotides_inv(new_s.seq[ii - 1]),
                    nucleotides_inv(new_s.seq[ii]),
                );
        }
        Likelihood::Scalar(
            (vector_proba * self.get_degenerate_end(nucleotides_inv(*new_s.seq.last().unwrap())))
                [(0, 0)],
        )
    }

    pub fn update_degenerate(&self, s: &Dna, first: usize, likelihood: f64) -> Array2<f64> {
        let mut transition_mat = Array2::zeros((4, 4));
        // Just ignore degenerate nucleotides (valid for N, which is the most common case).
        // For the other degenerate nucleotides, less important
        if !is_degenerate(s.seq[0]) {
            transition_mat[[first, nucleotides_inv(s.seq[0])]] += likelihood;
        }
        for ii in 1..s.len() {
            if !is_degenerate(s.seq[ii - 1]) && !is_degenerate(s.seq[ii]) {
                transition_mat[[nucleotides_inv(s.seq[ii - 1]), nucleotides_inv(s.seq[ii])]] +=
                    likelihood
            }
        }
        transition_mat
    }

    pub fn get_degenerate_matrix(&self, prev: usize, next: usize) -> Matrix4 {
        self.degenerate_matrix[next + 15 * prev]
    }

    pub fn get_degenerate_end(&self, nuc: usize) -> Vector4 {
        self.end_degenerate_vector[nuc]
    }

    pub fn precompute_degenerate(&mut self) {
        self.degenerate_matrix = vec![];
        // first precompute the transfer matrices
        for prev in NUCLEOTIDES {
            for next in NUCLEOTIDES {
                let mut matrix = Matrix4::new(
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                );

                for ii in 0..4 {
                    for jj in 0..4 {
                        if compatible_nucleotides(NUCLEOTIDES[ii], prev)
                            && compatible_nucleotides(NUCLEOTIDES[jj], next)
                        {
                            // transition matrix is P_{a,b} = P(a -> b)
                            matrix[(ii, jj)] = self.transition_matrix[[ii, jj]]
                        }
                    }
                }
                // println!("{:?}  {:?} \n {}  \n\n ", prev, next, matrix.to_string(),);
                self.degenerate_matrix.push(matrix);
            }
        }

        // then precompute the starts & ends:
        self.end_degenerate_vector = vec![];
        for nuc in NUCLEOTIDES {
            let mut vector_end = Vector4::new(0., 0., 0., 0.);
            for ii in 0..4 {
                if compatible_nucleotides(NUCLEOTIDES[ii], nuc) {
                    vector_end[ii] = 1.;
                }
            }
            self.end_degenerate_vector.push(vector_end);
        }
    }
}
/// functions specific to "amino-acid" dna sequences
impl DNAMarkovChain {
    pub fn likelihood_aminoacid(&self, s: &AminoAcid, start_chain: usize) -> Likelihood {
        Likelihood::Matrix(
            // if empty sequence
            if s.seq.len() == 0 || (s.seq.len() == 1 && s.start + s.end == 3) {
                Matrix16::identity()
            }
            // weird case #1, only one codon
            else if s.seq.len() == 1 {
                if self.reverse {
                    self.aa_lone_rev[&(s.seq[0], s.start, s.end, start_chain)]
                } else {
                    self.aa_lone[&(s.seq[0], s.start, s.end, start_chain)]
                }
            } else {
                // standard case
                if self.reverse {
                    let mut m = self.aa_start_rev[&(s.seq[s.seq.len() - 1], s.end, start_chain)];
                    for ii in (1..s.seq.len() - 1).rev() {
                        m *= self.aa_middle_rev[&s.seq[ii]];
                    }
                    let mf = m * self.aa_end_rev[&(s.seq[0], s.start)];
                    mf.transpose()
                } else {
                    let mut m = self.aa_start[&(s.seq[0], s.start, start_chain)];
                    for ii in 1..s.seq.len() - 1 {
                        m *= self.aa_middle[&s.seq[ii]]
                    }
                    let mf = m * self.aa_end[&(s.seq[s.seq.len() - 1], s.end)];
                    mf
                }
            },
        )
    }

    /// s is the sequence of interest and start_chain the 0th element.
    // pub fn likelihood_aminoacid(
    //     &self,
    //     s: &DegenerateCodonSequence,
    //     start_chain: usize,
    // ) -> Likelihood {
    //     if s.len() == 0 {
    //         return Likelihood::Matrix(Matrix16::identity());
    //     }

    //     let mut m = Matrix16::zeros();
    //     if !self.reverse {
    //         for (idx_left, idx_right, seq) in s.fix_extremities() {
    //             //                println!("{} {} {:?}", idx_left, idx_right, seq.to_dna());
    //             m[(idx_left, idx_right)] = self.likelihood_float_aa(&seq, start_chain);
    //         }
    //     } else {
    //         for (idx_left, idx_right, seq) in s.fix_extremities() {
    //             let mut new_seq = seq.clone();
    //             new_seq.reverse();
    //             // println!("{:?}", seq.to_dnas());
    //             // println!("{:?}", new_seq.to_dnas());
    //             m[(idx_left, idx_right)] = self.likelihood_float_aa(&new_seq, start_chain);
    //         }
    //     }
    //     Likelihood::Matrix(m)
    // }

    pub fn likelihood_float_aa(&self, s: &DegenerateCodonSequence, start_chain: usize) -> f64 {
        if s.codons.len() == 1 {
            return self.lonely_codon_likelihood(
                &s.codons[0],
                s.codon_start,
                s.codon_end,
                start_chain,
            );
        } else {
            // first codon
            let mut vector = self
                .start_codon_likelihood(&s.codons[0], s.codon_start, start_chain)
                .transpose()
                .clone();

            for codon in &s.codons[1..s.codons.len() - 1] {
                vector = vector * self.interior_codon_likelihood(codon);
            }

            // println!(
            //     "{} {:?}",
            //     s.to_dna(),
            //     (vector * self.end_codon_likelihood(s.codons.last().unwrap(), s.codon_end))[(0, 0)]
            // );

            // last codon, we know that there is at least one element in the sequence
            return (vector * self.end_codon_likelihood(s.codons.last().unwrap(), s.codon_end))
                [(0, 0)];
        }
    }

    pub fn update_aminoacid(&self, _s: &AminoAcid, _first: usize, _likelihood: f64) -> Array2<f64> {
        unimplemented!("Cannot update from an amino-acid sequence."); // for now (but maybe an error in the future)
    }

    /// Return a matrix A(τ, σ), where τ is the last nucleotide
    /// of the previous codon and σ is the last nucleotide
    pub fn interior_codon_likelihood(&self, codons: &DegenerateCodon) -> Matrix4 {
        // easiest and most frequent case, internal codon
        let mut matrix = Matrix4::new(
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        );

        for cod in &codons.triplets {
            for ii in 0..4 {
                matrix[(ii, cod[2])] += self.transition_matrix[[ii, cod[0]]]
                    * self.transition_matrix[[cod[0], cod[1]]]
                    * self.transition_matrix[[cod[1], cod[2]]];
            }
        }

        matrix
    }

    pub fn precompute_amino_acid(&mut self) {
        // this is a bit messy, to be clear with what's happening:
        // there's three type of matrices, S (16x4), M (4x4) and E (4x16),
        // each defined for each codon (and for S/E, where the codon starts)
        // S and E also depend on the "first nucleotide".
        // the "start" matrix is always the first in the "real sequence"
        // If the sequence is not! reversed (VD), then
        // the likelihood of the sequence is given by SxMxMxM...xMxE.
        // If the sequenc eis reversed (DJ), then the likelihood of
        // the sequence is given by (ExMxMx...xMxS).transpose()
        // TODO ^ is that true ?

        // precompute start codons
        self.aa_lone_rev = HashMap::new();
        self.aa_lone = HashMap::new();
        self.aa_start_rev = HashMap::new();
        self.aa_start = HashMap::new();
        self.aa_middle_rev = HashMap::new();
        self.aa_middle = HashMap::new();
        self.aa_end_rev = HashMap::new();
        self.aa_end = HashMap::new();

        if self.reverse {
            for codon in AMINOACIDS {
                for start_pos in 0..3 {
                    for previous_nucleotide in 0..4 {
                        self.aa_start_rev.insert(
                            (codon, start_pos, previous_nucleotide),
                            DegenerateCodon::from_amino(codon).reversed_start_codon_matrix(
                                self,
                                start_pos,
                                previous_nucleotide,
                            ),
                        );

                        for end_pos in 0..3 {
                            if start_pos + end_pos < 3 {
                                self.aa_lone_rev.insert(
                                    (codon, start_pos, end_pos, previous_nucleotide),
                                    DegenerateCodon::from_amino(codon)
                                        .reversed_lonely_codon_matrix(
                                            self,
                                            start_pos,
                                            end_pos,
                                            previous_nucleotide,
                                        ),
                                );
                            }
                        }
                    }
                }
                for end_pos in 0..3 {
                    self.aa_end_rev.insert(
                        (codon, end_pos),
                        DegenerateCodon::from_amino(codon).reversed_end_codon_matrix(self, end_pos),
                    );
                }
                self.aa_middle_rev.insert(
                    codon,
                    DegenerateCodon::from_amino(codon).reversed_middle_codon_matrix(self),
                );
            }
        } else {
            for codon in AMINOACIDS {
                for start_pos in 0..3 {
                    for previous_nucleotide in 0..4 {
                        self.aa_start.insert(
                            (codon, start_pos, previous_nucleotide),
                            DegenerateCodon::from_amino(codon).start_codon_matrix(
                                self,
                                start_pos,
                                previous_nucleotide,
                            ),
                        );

                        for end_pos in 0..3 {
                            if start_pos + end_pos < 3 {
                                self.aa_lone.insert(
                                    (codon, start_pos, end_pos, previous_nucleotide),
                                    DegenerateCodon::from_amino(codon).lonely_codon_matrix(
                                        self,
                                        start_pos,
                                        end_pos,
                                        previous_nucleotide,
                                    ),
                                );
                            }
                        }
                    }
                }
                for end_pos in 0..3 {
                    self.aa_end.insert(
                        (codon, end_pos),
                        DegenerateCodon::from_amino(codon).end_codon_matrix(self, end_pos),
                    );
                }

                self.aa_middle.insert(
                    codon,
                    DegenerateCodon::from_amino(codon).middle_codon_matrix(self),
                );
            }
        }
    }

    pub fn start_codon_likelihood(
        &self,
        codons: &DegenerateCodon,
        start: usize,
        first: usize,
    ) -> Vector4 {
        let mut vector = Vector4::new(0., 0., 0., 0.);
        if start == 0 {
            for cod in &codons.triplets {
                vector[cod[2]] += self.transition_matrix[[first, cod[0]]]
                    * self.transition_matrix[[cod[0], cod[1]]]
                    * self.transition_matrix[[cod[1], cod[2]]]
            }
        } else if start == 1 {
            for cod in &codons.triplets {
                //                if cod[0] == first {
                vector[cod[2]] += self.transition_matrix[[first, cod[1]]]
                    * self.transition_matrix[[cod[1], cod[2]]];
                //              }
            }
        } else if start == 2 {
            for cod in &codons.triplets {
                //                if cod[1] == first {
                vector[cod[2]] += self.transition_matrix[[first, cod[2]]];
                //              }
            }
        }
        vector
    }

    /// Return the likelihood vector p(ii, last)
    pub fn end_codon_likelihood(&self, codons: &DegenerateCodon, end: usize) -> Vector4 {
        let mut vector = Vector4::new(0., 0., 0., 0.);
        if end == 0 {
            for cod in &codons.triplets {
                for ii in 0..4 {
                    vector[ii] += self.transition_matrix[[ii, cod[0]]]
                        * self.transition_matrix[[cod[0], cod[1]]]
                        * self.transition_matrix[[cod[1], cod[2]]];
                }
            }
        } else if end == 1 {
            for cod in &codons.triplets {
                for ii in 0..4 {
                    vector[ii] += self.transition_matrix[[ii, cod[0]]]
                        * self.transition_matrix[[cod[0], cod[1]]];
                }
            }
        } else if end == 2 {
            for cod in &codons.triplets {
                for ii in 0..4 {
                    vector[ii] += self.transition_matrix[[ii, cod[0]]];
                }
            }
        }
        vector
    }

    /// Return the likelihood if the sequence is one codon long (special case)
    pub fn lonely_codon_likelihood(
        &self,
        codons: &DegenerateCodon,
        start: usize,
        end: usize,
        start_mc: usize,
    ) -> f64 {
        match (start, end) {
            (2, 1) => 1., // empty sequence
            (1, 2) => 1., // empty sequence
            (1, 1) => codons
                .triplets
                .iter()
                .map(|x| {
                    //                    if x[0] == start_mc {
                    self.transition_matrix[[start_mc, x[1]]]
                    // } else {
                    //     0.
                    // }
                })
                .sum::<f64>(),
            (2, 0) => codons
                .triplets
                .iter()
                .map(|x| {
                    //                    if x[1] == start_mc {
                    self.transition_matrix[[start_mc, x[2]]]
                    // } else {
                    //     0.
                    // }
                })
                .sum::<f64>(),
            (0, 2) => codons
                .triplets
                .iter()
                .map(|x| self.transition_matrix[[start_mc, x[0]]])
                .sum::<f64>(),
            (0, 1) => codons
                .triplets
                .iter()
                .map(|x| {
                    self.transition_matrix[[start_mc, x[0]]] * self.transition_matrix[[x[0], x[1]]]
                })
                .sum::<f64>(),
            (1, 0) => codons
                .triplets
                .iter()
                .map(|x| {
                    //if x[0] == start_mc {
                    self.transition_matrix[[start_mc, x[1]]] * self.transition_matrix[[x[1], x[2]]]
                    // } else {
                    //     0.
                    // }
                })
                .sum::<f64>(),
            (0, 0) => codons
                .triplets
                .iter()
                .map(|x| {
                    self.transition_matrix[[start_mc, x[0]]]
                        * self.transition_matrix[[x[0], x[1]]]
                        * self.transition_matrix[[x[1], x[2]]]
                })
                .sum::<f64>(),
            (_, _) => panic!("Wrong values in codon likelihood computation (likelihood_one_codon)"),
        }
    }
}
