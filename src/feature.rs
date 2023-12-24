use crate::utils::Normalize;
use crate::utils_sequences::{Dna, NUCLEOTIDES_INV};
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Array3, Axis};
use pyo3::pyclass;

// This class define different type of Feature
// Feature are used during the expectation maximization process
// In short you need:
// - a function computing the likelihood of the feature `likelihood`
// - a function that allows to update the probability distribution
//   when new observations are made.
//   This update is done lazily, for speed reason we don't want to
//   redefine the function everytime.
// - so a function cleanup is going to return a new object with
//   the new probability distribution.
// This is the general idea, there's quite a lot of boiler plate
// code for the different type of categorical features (categorical features in
// 1d, in 1d but given another parameter, in 2d ...)

pub trait Feature<T> {
    fn dirty_update(&mut self, observation: T, likelihood: f64);
    fn likelihood(&self, observation: T) -> f64;
    fn cleanup(&self) -> Result<Self>
    where
        Self: Sized;

    fn average(iter: impl Iterator<Item = Self> + ExactSizeIterator) -> Result<Self>
    where
        Self: Sized;
}

// One-dimensional categorical distribution
#[derive(Default, Clone, Debug)]
#[pyclass]
pub struct CategoricalFeature1 {
    pub probas: Array1<f64>,
    probas_dirty: Array1<f64>,
}

impl Feature<usize> for CategoricalFeature1 {
    fn dirty_update(&mut self, observation: usize, likelihood: f64) {
        self.probas_dirty[[observation]] += likelihood;
    }
    fn likelihood(&self, observation: usize) -> f64 {
        if observation >= self.probas.dim() {
            // TODO: not a fan of that, it's valid for the number of insertion
            // but clearly not for the V-gene for example, could hide errors...
            return 0.;
        }
        self.probas[[observation]]
    }
    fn cleanup(&self) -> Result<CategoricalFeature1> {
        CategoricalFeature1::new(&self.probas_dirty)
    }
    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature1> + ExactSizeIterator,
    ) -> Result<CategoricalFeature1> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas;
        for feat in iter {
            average_proba = average_proba + feat.probas;
            len += 1;
        }
        CategoricalFeature1::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature1 {
    pub fn new(probabilities: &Array1<f64>) -> Result<CategoricalFeature1> {
        Ok(CategoricalFeature1 {
            probas: probabilities.normalize_distribution(None)?,
            probas_dirty: Array1::<f64>::zeros(probabilities.dim()),
        })
    }
    pub fn dim(&self) -> usize {
        self.probas.dim()
    }
}

// One-dimensional categorical distribution, given one external parameter
#[derive(Default, Clone, Debug)]
#[pyclass]
pub struct CategoricalFeature1g1 {
    pub probas: Array2<f64>,
    probas_dirty: Array2<f64>,
}

impl Feature<(usize, usize)> for CategoricalFeature1g1 {
    fn dirty_update(&mut self, observation: (usize, usize), likelihood: f64) {
        self.probas_dirty[[observation.0, observation.1]] += likelihood;
    }
    fn likelihood(&self, observation: (usize, usize)) -> f64 {
        self.probas[[observation.0, observation.1]]
    }
    fn cleanup(&self) -> Result<CategoricalFeature1g1> {
        CategoricalFeature1g1::new(&self.probas_dirty)
    }

    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature1g1> + ExactSizeIterator,
    ) -> Result<CategoricalFeature1g1> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas;
        for feat in iter {
            average_proba = average_proba + feat.probas;
            len += 1;
        }
        CategoricalFeature1g1::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature1g1 {
    pub fn new(probabilities: &Array2<f64>) -> Result<CategoricalFeature1g1> {
        Ok(CategoricalFeature1g1 {
            probas: probabilities.normalize_distribution(Some(Axis(0)))?,
            probas_dirty: Array2::<f64>::zeros(probabilities.dim()),
        })
    }
    pub fn dim(&self) -> (usize, usize) {
        self.probas.dim()
    }
}

// Two-dimensional categorical distribution
#[derive(Default, Clone, Debug)]
#[pyclass]
pub struct CategoricalFeature2 {
    pub probas: Array2<f64>,
    probas_dirty: Array2<f64>,
}

impl Feature<(usize, usize)> for CategoricalFeature2 {
    fn dirty_update(&mut self, observation: (usize, usize), likelihood: f64) {
        self.probas_dirty[[observation.0, observation.1]] += likelihood;
    }
    fn likelihood(&self, observation: (usize, usize)) -> f64 {
        self.probas[[observation.0, observation.1]]
    }
    fn cleanup(&self) -> Result<CategoricalFeature2> {
        CategoricalFeature2::new(&self.probas_dirty)
    }
    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature2> + ExactSizeIterator,
    ) -> Result<CategoricalFeature2> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas;
        for feat in iter {
            average_proba = average_proba + feat.probas;
            len += 1;
        }
        CategoricalFeature2::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature2 {
    pub fn new(probabilities: &Array2<f64>) -> Result<CategoricalFeature2> {
        Ok(CategoricalFeature2 {
            probas: probabilities
                .normalize_distribution(Some(Axis(0)))?
                .normalize_distribution(Some(Axis(1)))?,
            probas_dirty: Array2::<f64>::zeros(probabilities.dim()),
        })
    }
    pub fn dim(&self) -> (usize, usize) {
        self.probas.dim()
    }
}

// Two-dimensional categorical distribution, given one external parameter
#[derive(Default, Clone, Debug)]
#[pyclass]
pub struct CategoricalFeature2g1 {
    pub probas: Array3<f64>,
    probas_dirty: Array3<f64>,
}

impl Feature<(usize, usize, usize)> for CategoricalFeature2g1 {
    fn dirty_update(&mut self, observation: (usize, usize, usize), likelihood: f64) {
        self.probas_dirty[[observation.0, observation.1, observation.2]] += likelihood;
    }
    fn likelihood(&self, observation: (usize, usize, usize)) -> f64 {
        self.probas[[observation.0, observation.1, observation.2]]
    }
    fn cleanup(&self) -> Result<CategoricalFeature2g1> {
        CategoricalFeature2g1::new(&self.probas_dirty)
    }
    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature2g1> + ExactSizeIterator,
    ) -> Result<CategoricalFeature2g1> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas;
        for feat in iter {
            average_proba = average_proba + feat.probas;
            len += 1;
        }
        CategoricalFeature2g1::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature2g1 {
    pub fn new(probabilities: &Array3<f64>) -> Result<CategoricalFeature2g1> {
        Ok(CategoricalFeature2g1 {
            probas: probabilities
                .normalize_distribution(Some(Axis(0)))?
                .normalize_distribution(Some(Axis(1)))?,
            probas_dirty: Array3::<f64>::zeros(probabilities.dim()),
        })
    }
    pub fn dim(&self) -> (usize, usize, usize) {
        self.probas.dim()
    }
}

// Markov chain structure for Dna insertion
#[derive(Default, Clone, Debug)]
#[pyclass]
pub struct MarkovFeature {
    pub initial_distribution: Array1<f64>,
    initial_distribution_internal: Array1<f64>,
    pub transition_matrix: Array2<f64>,
    transition_matrix_internal: Array2<f64>,
    initial_distribution_dirty: Array1<f64>,
    transition_matrix_dirty: Array2<f64>,
}

impl Feature<&Dna> for MarkovFeature {
    fn dirty_update(&mut self, observation: &Dna, likelihood: f64) {
        if observation.is_empty() {
            return;
        }
        self.initial_distribution_dirty[[NUCLEOTIDES_INV[&observation.seq[0]]]] += likelihood;
        for ii in 1..observation.len() {
            if (observation.seq[ii - 1] != b'N') & (observation.seq[ii] != b'N') {
                self.transition_matrix_dirty[[
                    NUCLEOTIDES_INV[&observation.seq[ii - 1]],
                    NUCLEOTIDES_INV[&observation.seq[ii]],
                ]] += likelihood;
            }
        }
    }
    fn likelihood(&self, observation: &Dna) -> f64 {
        if observation.is_empty() {
            return 1.;
        }
        let mut proba = self.initial_distribution_internal[NUCLEOTIDES_INV[&observation.seq[0]]];
        for ii in 1..observation.len() {
            proba *= self.transition_matrix_internal[[
                NUCLEOTIDES_INV[&observation.seq[ii - 1]],
                NUCLEOTIDES_INV[&observation.seq[ii]],
            ]];
        }
        proba
    }
    fn cleanup(&self) -> Result<MarkovFeature> {
        MarkovFeature::new(&self.initial_distribution, &self.transition_matrix)
    }
    fn average(
        mut iter: impl Iterator<Item = MarkovFeature> + ExactSizeIterator,
    ) -> Result<MarkovFeature> {
        let mut len = 1;
        let first_feat = iter.next().ok_or(anyhow!("Cannot average empty vector"))?;
        let mut average_init = first_feat.initial_distribution;
        let mut average_mat = first_feat.transition_matrix;
        for feat in iter {
            average_init = average_init + feat.initial_distribution;
            average_mat = average_mat + feat.transition_matrix;
            len += 1;
        }
        MarkovFeature::new(
            &(average_init / (len as f64)),
            &(average_mat / (len as f64)),
        )
    }
}

impl MarkovFeature {
    pub fn new(
        initial_distribution: &Array1<f64>,
        transition_matrix: &Array2<f64>,
    ) -> Result<MarkovFeature> {
        let mut m = MarkovFeature {
            transition_matrix: transition_matrix.normalize_distribution(Some(Axis(1)))?,
            initial_distribution: initial_distribution.normalize_distribution(None)?,
            transition_matrix_dirty: Array2::<f64>::zeros(transition_matrix.dim()),
            initial_distribution_dirty: Array1::<f64>::zeros(initial_distribution.dim()),
            transition_matrix_internal: Array2::<f64>::zeros((5, 5)),
            initial_distribution_internal: Array1::<f64>::zeros(5),
        };
        m.add_undefined_nucleotides();
        Ok(m)
    }
    pub fn get_parameters(&self) -> (Array1<f64>, Array2<f64>) {
        (
            self.initial_distribution.clone(),
            self.transition_matrix.clone(),
        )
    }

    fn add_undefined_nucleotides(&mut self) {
        // deal with undefined -N- nucleotides
        for ii in 0..4 {
            self.initial_distribution_internal[ii] = self.initial_distribution[ii];
            for jj in 0..4 {
                self.transition_matrix_internal[[ii, jj]] = self.transition_matrix[[ii, jj]];
            }
        }
        self.initial_distribution_internal[4] = 1.;
        for ii in 0..5 {
            self.transition_matrix_internal[[ii, 4]] = 1.;
            self.transition_matrix_internal[[4, ii]] = 1.;
        }
    }
}

// Most basic error model
#[derive(Default, Clone, Debug)]
#[pyclass]
pub struct ErrorPoisson {
    #[pyo3(get, set)]
    pub error_rate: f64,
    lookup_table: Vec<f64>,
    total_probas_dirty: f64, // useful for dirty updating
    total_errors_dirty: f64, // same
    min_likelihood: f64,     // event with a lower likelihood are ignored
}

fn make_lookup_table(error_rate: f64, min_likelihood: f64) -> Vec<f64> {
    let mut lookup_table = Vec::<f64>::new();
    let mut prob = (-error_rate).exp();
    let mut nb = 0;
    loop {
        lookup_table.push(prob);
        nb += 1;
        prob *= error_rate / (nb as f64);
        if prob < min_likelihood {
            break;
        }
    }
    lookup_table
}

impl ErrorPoisson {
    pub fn new(error_rate: f64, min_likelihood: f64) -> Result<ErrorPoisson> {
        if (error_rate < 0.) | (error_rate.is_nan()) | (error_rate.is_infinite()) {
            return Err(anyhow!(
                "Error in ErrorPoisson Feature creation. Negative/NaN/infinite error rate."
            ));
        }
        if min_likelihood < 0. {
            return Err(anyhow!(
                "Error in ErrorPoisson Feature creation. Negative min likelihood."
            ));
        }

        Ok(ErrorPoisson {
            error_rate,
            min_likelihood,
            total_probas_dirty: 0.,
            total_errors_dirty: 0.,
            lookup_table: make_lookup_table(error_rate, min_likelihood),
        })
    }
}

impl Feature<usize> for ErrorPoisson {
    fn dirty_update(&mut self, observation: usize, likelihood: f64) {
        self.total_probas_dirty += likelihood;
        self.total_errors_dirty += likelihood * (observation as f64);
    }
    fn likelihood(&self, observation: usize) -> f64 {
        if observation >= self.lookup_table.len() {
            0.
        } else {
            self.lookup_table[observation]
        }
    }
    fn cleanup(&self) -> Result<ErrorPoisson> {
        // estimate the error_rate from the dirty estimates
        // The new mean is sum(nb * proba)/sum(proba)
        // which is the MLE of the Poisson distribution

        if self.total_errors_dirty < self.min_likelihood {
            return ErrorPoisson::new(0., self.min_likelihood);
        }
        ErrorPoisson::new(
            self.total_errors_dirty / self.total_probas_dirty,
            self.min_likelihood,
        )
    }
    fn average(
        mut iter: impl Iterator<Item = ErrorPoisson> + ExactSizeIterator,
    ) -> Result<ErrorPoisson> {
        let mut len = 1;
        let first_feat = iter.next().ok_or(anyhow!("Cannot average empty vector"))?;
        let mut average_err = first_feat.error_rate;
        let min_likelihood = first_feat.min_likelihood;
        for feat in iter {
            average_err += feat.error_rate;
            len += 1;
        }
        ErrorPoisson::new(average_err / (len as f64), min_likelihood)
    }
}
