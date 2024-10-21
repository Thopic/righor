// Deal with the subtleties of the Markov chain on Dna or Amino-Acid structure

//use crate::shared::distributions::calc_steady_state_dist;
use crate::shared::sequence::DnaLikeEnum;
use crate::shared::sequence::{compatible_nucleotides, is_degenerate, NUCLEOTIDES};
use crate::shared::{
    nucleotides_inv, sequence::DegenerateCodon, sequence::DegenerateCodonSequence,
    utils::normalize_transition_matrix, Dna, DnaLike,
};
use anyhow::Result;
use nalgebra::{Matrix4, Vector4};
use ndarray::Array2;

#[derive(Clone, PartialEq, Debug, Default)]
pub struct DNAMarkovChain {
    pub transition_matrix: Array2<f64>,
    // pre-computed data for dealing with the degenerate nucleotides
    pub degenerate_matrix: Vec<Matrix4<f64>>, // likelihood matrix for the degenerate dna
    pub end_degenerate_vector: Vec<Vector4<f64>>,
}

impl DNAMarkovChain {
    pub fn new(transition_matrix: &Array2<f64>) -> Result<DNAMarkovChain> {
        let mut mc = DNAMarkovChain::default();
        mc.precompute(transition_matrix)?;
        Ok(mc)
    }

    pub fn likelihood(&self, sequence: &DnaLike, first_nucleotide: usize) -> f64 {
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
        Ok(())
    }
}

// functions specific to non-degenerate dna sequences
impl DNAMarkovChain {
    pub fn likelihood_dna(&self, s: &Dna, first: usize) -> f64 {
        if s.len() == 0 {
            return 1.;
        }

        let mut proba = self.transition_matrix[[first, nucleotides_inv(s.seq[0])]];
        for ii in 1..s.len() {
            proba *= self.transition_matrix
                [[nucleotides_inv(s.seq[ii - 1]), nucleotides_inv(s.seq[ii])]];
        }
        proba
    }

    pub fn update_dna(&self, s: &Dna, first: usize, likelihood: f64) -> Array2<f64> {
        let mut transition_mat = Array2::zeros((4, 4));
        transition_mat[[first, nucleotides_inv(s.seq[0])]] += likelihood;
        for ii in 1..s.len() {
            transition_mat[[nucleotides_inv(s.seq[ii - 1]), nucleotides_inv(s.seq[ii])]] +=
                likelihood
        }
        transition_mat
    }
}

// functions specific to degenerate dna sequences
impl DNAMarkovChain {
    pub fn likelihood_degenerate(&self, s: &Dna, first: usize) -> f64 {
        if s.len() == 0 {
            return 1.;
        }

        let mut vector_proba = Vector4::new(0., 0., 0., 0.);
        vector_proba[first] = 1.;
        let mut vector_proba =
            vector_proba.transpose() * self.get_degenerate_matrix(first, nucleotides_inv(s.seq[0]));
        for ii in 1..s.len() {
            vector_proba = vector_proba
                * self.get_degenerate_matrix(
                    nucleotides_inv(s.seq[ii - 1]),
                    nucleotides_inv(s.seq[ii]),
                );
        }
        (vector_proba * self.get_degenerate_end(nucleotides_inv(*s.seq.last().unwrap())))[(0, 0)]
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

    pub fn get_degenerate_matrix(&self, prev: usize, next: usize) -> Matrix4<f64> {
        self.degenerate_matrix[next + 15 * prev]
    }

    pub fn get_degenerate_end(&self, nuc: usize) -> Vector4<f64> {
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
    /// s is the sequence of interest + its first element.
    pub fn likelihood_aminoacid(&self, s: &DegenerateCodonSequence, first: usize) -> f64 {
        if s.len() == 0 {
            return 1.;
        }

        if s.codons.len() == 1 {
            return self.lonely_codon_likelihood(&s.codons[0], s.codon_start, s.codon_end, first);
        }

        // first codon
        let mut vector = self
            .start_codon_likelihood(&s.codons[0], s.codon_start, first)
            .transpose()
            .clone();

        for codon in &s.codons[1..s.codons.len() - 1] {
            vector = vector * self.interior_codon_likelihood(codon);
        }

        // last codon, we know that there is at least one element in the sequence
        (vector * self.end_codon_likelihood(s.codons.last().unwrap(), s.codon_end))[(0, 0)]
    }

    pub fn update_aminoacid(
        &self,
        _s: &DegenerateCodonSequence,
        _first: usize,
        _likelihood: f64,
    ) -> Array2<f64> {
        unimplemented!("Cannot update from an amino-acid sequence."); // for now (but maybe an error in the future)
    }

    /// Return a matrix A(τ, σ), where τ is the last nucleotide of the previous codon and
    /// σ is the last nucleotide
    pub fn interior_codon_likelihood(&self, codons: &DegenerateCodon) -> Matrix4<f64> {
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

    pub fn start_codon_likelihood(
        &self,
        codons: &DegenerateCodon,
        start: usize,
        first: usize,
    ) -> Vector4<f64> {
        let mut vector = Vector4::new(0., 0., 0., 0.);
        if start == 0 {
            for cod in &codons.triplets {
                vector[cod[2]] += self.transition_matrix[[first, cod[0]]]
                    * self.transition_matrix[[cod[0], cod[1]]]
                    * self.transition_matrix[[cod[1], cod[2]]]
            }
        } else if start == 1 {
            for cod in &codons.triplets {
                if cod[0] == first {
                    vector[cod[2]] += self.transition_matrix[[first, cod[1]]]
                        * self.transition_matrix[[cod[1], cod[2]]];
                }
            }
        } else if start == 2 {
            for cod in &codons.triplets {
                if cod[1] == first {
                    vector[cod[2]] += self.transition_matrix[[first, cod[2]]];
                }
            }
        }
        vector
    }

    /// Return the likelihood vector p(ii, last)
    pub fn end_codon_likelihood(&self, codons: &DegenerateCodon, end: usize) -> Vector4<f64> {
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
        first: usize, // the first nucleotide, 0/1/2/3
    ) -> f64 {
        match (start, end) {
            (2, 1) => 1., // empty sequence
            (1, 2) => 1., // empty sequence
            (1, 1) => codons
                .triplets
                .iter()
                .map(|x| {
                    if x[0] == first {
                        self.transition_matrix[[x[0], x[1]]]
                    } else {
                        0.
                    }
                })
                .sum::<f64>(),
            (2, 0) => codons
                .triplets
                .iter()
                .map(|x| {
                    if x[1] == first {
                        self.transition_matrix[[x[1], x[2]]]
                    } else {
                        0.
                    }
                })
                .sum::<f64>(),
            (0, 2) => codons
                .triplets
                .iter()
                .map(|x| self.transition_matrix[[first, x[0]]])
                .sum::<f64>(),
            (0, 1) => codons
                .triplets
                .iter()
                .map(|x| {
                    self.transition_matrix[[first, x[0]]] * self.transition_matrix[[x[0], x[1]]]
                })
                .sum::<f64>(),
            (1, 0) => codons
                .triplets
                .iter()
                .map(|x| {
                    if x[0] == first {
                        self.transition_matrix[[x[0], x[1]]] * self.transition_matrix[[x[1], x[2]]]
                    } else {
                        0.
                    }
                })
                .sum::<f64>(),
            (0, 0) => codons
                .triplets
                .iter()
                .map(|x| {
                    self.transition_matrix[[first, x[0]]]
                        * self.transition_matrix[[x[0], x[1]]]
                        * self.transition_matrix[[x[1], x[2]]]
                })
                .sum::<f64>(),
            (_, _) => panic!("Wrong values in codon likelihood computation (likelihood_one_codon)"),
        }
    }
}
