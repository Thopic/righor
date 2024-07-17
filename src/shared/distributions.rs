//! Distributions to improve the speed of the generation process
use crate::shared::sequence::{nucleotides_inv, NUCLEOTIDES};
use crate::shared::utils::normalize_transition_matrix;
use crate::shared::Dna;
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Uniform, WeightedAliasIndex};

#[derive(Clone, Debug)]
/// Pick a random number according to a histogram defined distribution
pub struct HistogramDistribution {
    bin_pick: DiscreteDistribution,
    uniform_in_bins: Vec<Uniform<f64>>,
}

impl HistogramDistribution {
    pub fn new(bins: Vec<f64>, probas: Vec<f64>) -> Result<HistogramDistribution> {
        Ok(HistogramDistribution {
            bin_pick: DiscreteDistribution::new(probas)?,
            uniform_in_bins: bins
                .windows(2)
                .map(|w| Uniform::try_from(w[0]..w[1]).unwrap())
                .collect(),
        })
    }

    pub fn generate<R: Rng>(&self, rng: &mut R) -> f64 {
        let idx = self.bin_pick.generate(rng);
        self.uniform_in_bins[idx].sample(rng)
    }
}

impl Default for HistogramDistribution {
    fn default() -> HistogramDistribution {
        HistogramDistribution::new(vec![0., 1e-12], vec![1.]).unwrap()
    }
}

#[derive(Clone, Debug)]
/// Generate a random f64 between 0/1 and a random nucleotide
pub struct UniformError {
    is_error: Uniform<f64>,
    nucleotide: Uniform<usize>,
}

impl Default for UniformError {
    fn default() -> UniformError {
        UniformError::new()
    }
}

impl UniformError {
    pub fn new() -> UniformError {
        UniformError {
            is_error: Uniform::new(0.0, 1.0),
            nucleotide: Uniform::new_inclusive(0, 3),
        }
    }

    pub fn is_error<R: Rng>(&self, er: f64, rng: &mut R) -> bool {
        self.is_error.sample(rng) < er
    }

    pub fn random_nucleotide<R: Rng>(&self, rng: &mut R) -> u8 {
        NUCLEOTIDES[self.nucleotide.sample(rng)]
    }
}

/// Generate an integer with a given probability
#[derive(Clone, Debug)]
pub struct DiscreteDistribution {
    distribution: WeightedAliasIndex<f64>,
}

impl DiscreteDistribution {
    pub fn new(weights: Vec<f64>) -> Result<Self> {
        if !weights.iter().all(|&x| x >= 0.) {
            return Err(anyhow!(
                "Error when creating distribution: negative weights"
            ))?;
        }

        let distribution = match weights.iter().sum::<f64>().abs() < 1e-10 {
	    true => WeightedAliasIndex::new(vec![1.; weights.len()]) // when all the value are 0, all the values are equiprobable.
		.map_err(|e| anyhow!(format!("Error when creating distribution: {}", e)))?,
	    false => WeightedAliasIndex::new(weights)
		.map_err(|e| anyhow!(format!("Error when creating distribution: {}", e)))?
	};
        Ok(DiscreteDistribution { distribution })
    }

    pub fn generate<R: Rng>(&self, rng: &mut R) -> usize {
        self.distribution.sample(rng)
    }
}

impl Default for DiscreteDistribution {
    fn default() -> Self {
        DiscreteDistribution {
            distribution: WeightedAliasIndex::new(vec![1.]).unwrap(),
        }
    }
}

// Error model
#[derive(Clone, Debug)]
pub struct ErrorDistribution {
    pub is_error: Uniform<f64>,
    pub nucleotide: Uniform<usize>,
}

impl Default for ErrorDistribution {
    fn default() -> Self {
        ErrorDistribution {
            is_error: Uniform::new(0.0, 1.0),
            nucleotide: Uniform::new_inclusive(0, 3),
        }
    }
}

// Markov chain structure (for the insertion process)
#[derive(Default, Clone, Debug)]
pub struct MarkovDNA {
    // initial_distribution: DiscreteDistribution, // first nucleotide, ACGT order
    transition_matrix: Vec<DiscreteDistribution>, // Markov matrix, ACGT order
}

impl MarkovDNA {
    pub fn new(transition_probs: Array2<f64>) -> Result<Self> {
        let mut transition_matrix = Vec::with_capacity(transition_probs.dim().0);
        for probs in transition_probs.axis_iter(Axis(0)) {
            transition_matrix.push(DiscreteDistribution::new(probs.to_vec())?);
        }
        Ok(MarkovDNA { transition_matrix })
    }

    pub fn generate<R: Rng>(&mut self, length: usize, previous_nucleotide: u8, rng: &mut R) -> Dna {
        let mut dna = Dna {
            seq: Vec::with_capacity(length),
        };
        let mut current_state = nucleotides_inv(previous_nucleotide);
        for _ in 0..length {
            current_state = self.transition_matrix[current_state].generate(rng);
            dna.seq.push(NUCLEOTIDES[current_state]);
        }
        dna
    }
}

pub fn calc_steady_state_dist(transition_matrix: &Array2<f64>) -> Result<Vec<f64>> {
    // this should be profondly modified TODO
    // originally computed the eigenvalues. This is a pain though, because
    // it means I need to load blas, which takes forever to compile.
    // And this is not exactly an important part of the program.
    // so this means I'm going to do it stupidly

    // first normalize the transition matrix
    let mat = normalize_transition_matrix(transition_matrix)?;
    let epsilon: f64 = 1e-10;

    if mat.sum() == 0.0 {
        return Ok(vec![0.; mat.dim().0]);
    }

    let n = mat.nrows();
    let mut vec = Array1::from_elem(n, 1.0 / n as f64);
    for _ in 0..10000 {
        let vec_next = mat.dot(&vec);
        let norm = vec_next.sum();
        let vec_next = vec_next / norm;

        if (&vec_next - &vec).mapv(|a| a.abs()).sum() < epsilon {
            return Ok(vec_next.to_vec());
        }
        vec = vec_next;
    }
    Err(anyhow!("No suitable eigenvector found"))?
}
