use crate::utils_sequences::{Dna, NUCLEOTIDES, NUCLEOTIDES_INV};
use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_linalg::Eig;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rand_distr::WeightedAliasIndex;
use std::error::Error;

const EPSILON: f64 = 1e-10;

// Define some storage wrapper for the V/D/J genes

#[derive(Default, Clone, Debug)]
pub struct Gene {
    pub name: String,
    pub seq: Dna,
    pub seq_with_pal: Option<Dna>, // Dna with the palindromic insertions (model dependant)
    pub functional: String,
    pub cdr3_pos: Option<usize>, // start (for V gene) or end (for J gene) of CDR3
}

impl Gene {
    pub fn create_palindromic_ends(&mut self, lenleft: usize, lenright: usize) {
        let palindromic_extension_left = self
            .seq
            .extract_subsequence(0, lenleft)
            .reverse_complement();
        let mut seqpal: Vec<u8> = palindromic_extension_left
            .seq
            .into_iter()
            .chain(self.seq.seq.clone().into_iter())
            .collect();
        let palindromic_extension_right = self
            .seq
            .extract_subsequence(self.seq.len() - lenright, self.seq.len())
            .reverse_complement();
        seqpal.extend(palindromic_extension_right.seq);

        self.seq_with_pal = Some(Dna {
            seq: seqpal.clone(),
        });
    }
}

// Define the random distributions used in the rest of the code.

// Generate an integer with a given probability
#[derive(Clone, Debug)]
pub struct DiscreteDistribution {
    distribution: WeightedAliasIndex<f64>,
}

impl DiscreteDistribution {
    pub fn new(weights: Vec<f64>) -> Result<Self, Box<dyn Error>> {
        if !weights.iter().all(|&x| x >= 0.) {
            return Err("Error when creating distribution: negative weights")?;
        }

        let distribution = match weights.iter().sum::<f64>().abs() < 1e-10 {
	    true => WeightedAliasIndex::new(vec![1.; weights.len()]) // when all the value are 0, all the values are equiprobable.
		.map_err(|e| format!("Error when creating distribution: {}", e))?,
	    false => WeightedAliasIndex::new(weights)
		.map_err(|e| format!("Error when creating distribution: {}", e))?
	};
        Ok(DiscreteDistribution {
            distribution: distribution,
        })
    }

    pub fn generate<R: Rng>(&mut self, rng: &mut R) -> usize {
        self.distribution.sample(rng)
    }
}

impl Default for DiscreteDistribution {
    fn default() -> Self {
        return DiscreteDistribution {
            distribution: WeightedAliasIndex::new(vec![1.]).unwrap(),
        };
    }
}

// Markov chain structure (for the insertion process)
#[derive(Default, Clone, Debug)]
pub struct MarkovDNA {
    initial_distribution: DiscreteDistribution, // first nucleotide, ACGT order
    transition_matrix: Vec<DiscreteDistribution>, // Markov matrix, ACGT order
}

impl MarkovDNA {
    pub fn new(
        transition_probs: Array2<f64>,
        initial_probs: Option<Vec<f64>>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut transition_matrix = Vec::with_capacity(transition_probs.dim().0);
        for probs in transition_probs.axis_iter(Axis(0)) {
            transition_matrix.push(DiscreteDistribution::new(probs.to_vec())?);
        }
        let initial_distribution = match initial_probs {
            None => DiscreteDistribution::new(calc_steady_state_dist(&transition_probs)?)?,
            Some(dist) => DiscreteDistribution::new(dist)?,
        };
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

pub fn calc_steady_state_dist(transition_matrix: &Array2<f64>) -> Result<Vec<f64>, Box<dyn Error>> {
    let (eig, eigv) = transition_matrix
        .eig()
        .map_err(|_| "Eigen decomposition failed")?;
    for i in 0..4 {
        if (eig[i].re - 1.).abs() < 1e-6 {
            let col = eigv.column(i);
            let sum: f64 = col.mapv(|x| x.re).sum();
            return Ok(col.mapv(|x| x.re / sum).to_vec());
        }
    }
    Err("No suitable eigenvector found")?
}

pub fn likelihood_markov(
    initial_distribution: &Array1<f64>,
    transition_matrix: &Array2<f64>,
    d: &Dna,
) -> f64 {
    // estimate the log likelihood of a specific Dna sequence in a Markov chain model
    if d.len() == 0 {
        return 0.;
    }
    let mut proba = initial_distribution[NUCLEOTIDES_INV[&d.seq[0]]];
    for ii in 1..d.len() {
        proba *= transition_matrix[[NUCLEOTIDES_INV[&d.seq[ii - 1]], NUCLEOTIDES_INV[&d.seq[ii]]]];
    }
    proba
}

pub fn update_markov_probas(
    initial_distribution: &mut Array1<f64>,
    transition_matrix: &mut Array2<f64>,
    d: &Dna,
    likelihood: f64,
) {
    if d.len() == 0 {
        return;
    }
    initial_distribution[[NUCLEOTIDES_INV[&d.seq[0]]]] += likelihood;
    for ii in 1..d.len() {
        transition_matrix[[NUCLEOTIDES_INV[&d.seq[ii - 1]], NUCLEOTIDES_INV[&d.seq[ii]]]] +=
            likelihood;
    }
}

pub fn add_errors<R: Rng>(dna: &mut Dna, error_rate: f64, rng: &mut R) {
    let uniform = Uniform::new(0.0, 1.0);
    let random_nucleotide = Uniform::new_inclusive(0, 3);

    for nucleotide in dna.seq.iter_mut() {
        if uniform.sample(rng) < error_rate {
            *nucleotide = NUCLEOTIDES[random_nucleotide.sample(rng)];
        }
    }
}

pub trait Normalize {
    fn normalize_distribution(&self, axis: Option<Axis>) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;
}

impl Normalize for Array1<f64> {
    fn normalize_distribution(&self, _axis: Option<Axis>) -> Result<Self, Box<dyn Error>> {
        if self.iter().any(|&x| x < 0.0) {
            // negative values mean something wrong happened
            return Err("Array contains non-positive values".into());
        }

        let sum = self.sum();
        if sum.abs() < EPSILON {
            // return a uniform distribution
            return Ok(Array1::ones(self.dim()) / self.dim() as f64);
        }

        Ok(self / sum)
    }
}

impl Normalize for Array2<f64> {
    fn normalize_distribution(&self, axis: Option<Axis>) -> Result<Self, Box<dyn Error>> {
        if self.iter().any(|&x| !x.is_finite()) {
            return Err("Array contains non-positive or non-finite values".into());
        }

        match axis {
            Some(ax) => {
                let mut normalized = self.clone();
                let sums = self.sum_axis(ax);
                for (mut slice, sum) in normalized.axis_iter_mut(ax).zip(sums.iter()) {
                    if sum.abs() < EPSILON {
                        // if the row sums to 0. we return an uniform probability distribution
                        slice.mapv_inplace(|_| 1.0 / self.len_of(ax) as f64)
                    } else {
                        slice.mapv_inplace(|x| x / sum);
                    }
                }
                return Ok(normalized);
            }
            None => return Err("Unspecified axis")?,
        }
    }
}

impl Normalize for Array3<f64> {
    fn normalize_distribution(&self, axis: Option<Axis>) -> Result<Self, Box<dyn Error>> {
        if self.iter().any(|&x| !x.is_finite()) {
            return Err("Array contains non-positive or non-finite values".into());
        }

        match axis {
            Some(ax) => {
                let mut normalized = self.clone();
                let sums = self.sum_axis(ax);
                for (mut slice, sum) in normalized.axis_iter_mut(ax).zip(sums.iter()) {
                    if sum.abs() < EPSILON {
                        // if the row sums to 0. we return an uniform probability distribution
                        slice.mapv_inplace(|_| 1.0 / self.len_of(ax) as f64)
                    } else {
                        slice.mapv_inplace(|x| x / sum);
                    }
                }
                return Ok(normalized);
            }
            None => return Err("Unspecified axis")?,
        }
    }
}
