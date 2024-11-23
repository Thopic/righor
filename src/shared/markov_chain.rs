// Deal with the subtleties of the Markov chain on Dna or Amino-Acid structure

//use crate::shared::distributions::calc_steady_state_dist;
use crate::shared::likelihood::Likelihood;
use crate::shared::likelihood::{Matrix16, Matrix16x4, Matrix4, Matrix4x16, Vector4};
use crate::shared::sequence::{
    compatible_nucleotides, is_degenerate, ALL_POSSIBLE_CODONS_AA, NUCLEOTIDES,
};
use crate::shared::sequence::{AminoAcid, Dna, DnaLike, DnaLikeEnum};
use crate::shared::{
    amino_acids::DegenerateCodon, nucleotides_inv, utils::normalize_transition_matrix,
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
                Box::new(Matrix16::identity())
            }
            // weird case #1, only one codon
            else if s.seq.len() == 1 {
                if self.reverse {
                    Box::new(self.aa_lone_rev[&(s.seq[0], s.start, s.end, start_chain)])
                } else {
                    Box::new(self.aa_lone[&(s.seq[0], s.start, s.end, start_chain)])
                }
            } else {
                // standard case
                if self.reverse {
                    let mut m_4_4 = Matrix4::identity();
                    for ii in (1..s.seq.len() - 1).rev() {
                        m_4_4 *= self.aa_middle_rev[&s.seq[ii]];
                    }

                    let m_16_4 =
                        self.aa_start_rev[&(s.seq[s.seq.len() - 1], s.end, start_chain)] * m_4_4;
                    let mf = m_16_4 * self.aa_end_rev[&(s.seq[0], s.start)];
                    Box::new(mf.transpose())
                } else {
                    let mut m_4_4 = Matrix4::identity();
                    for ii in 1..s.seq.len() - 1 {
                        m_4_4 *= self.aa_middle[&s.seq[ii]]
                    }
                    let m0 = self.aa_start[&(s.seq[0], s.start, start_chain)];
                    let mf = m0 * m_4_4 * self.aa_end[&(s.seq[s.seq.len() - 1], s.end)];
                    Box::new(mf)
                }
            },
        )
    }

    pub fn update_aminoacid(&self, _s: &AminoAcid, _first: usize, _likelihood: f64) -> Array2<f64> {
        unimplemented!("Cannot update from an amino-acid sequence."); // for now (but maybe an error in the future)
    }

    /// Precompute the matrices needed to compute the insertion probability
    /// in the amino-acid case.
    pub fn precompute_amino_acid(&mut self) {
        self.aa_lone_rev = HashMap::new();
        self.aa_lone = HashMap::new();
        self.aa_start_rev = HashMap::new();
        self.aa_start = HashMap::new();
        self.aa_middle_rev = HashMap::new();
        self.aa_middle = HashMap::new();
        self.aa_end_rev = HashMap::new();
        self.aa_end = HashMap::new();

        if self.reverse {
            for codon in ALL_POSSIBLE_CODONS_AA {
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
            for codon in ALL_POSSIBLE_CODONS_AA {
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
}
