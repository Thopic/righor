use crate::sequence::utils::{nucleotides_inv, Dna};
use crate::shared::utils::{normalize_transition_matrix, Normalize, Normalize2};
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Array3};
#[cfg(all(feature = "py_binds", feature = "py_o3"))]
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
    fn log_likelihood(&self, observation: T) -> f64;
    fn cleanup(&self) -> Result<Self>
    where
        Self: Sized;

    fn average(iter: impl Iterator<Item = Self> + ExactSizeIterator) -> Result<Self>
    where
        Self: Sized;
}

// One-dimensional categorical distribution
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pyclass)]
pub struct CategoricalFeature1 {
    pub log_probas: Array1<f64>,
    pub probas_dirty: Array1<f64>,
}

impl Feature<usize> for CategoricalFeature1 {
    fn dirty_update(&mut self, observation: usize, likelihood: f64) {
        self.probas_dirty[[observation]] += likelihood;
    }
    fn log_likelihood(&self, observation: usize) -> f64 {
        self.log_probas[[observation]]
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
            .log_probas
            .mapv(|x| x.exp2());
        for feat in iter {
            average_proba = average_proba + feat.log_probas.mapv(|x| x.exp2());
            len += 1;
        }
        CategoricalFeature1::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature1 {
    pub fn new(probabilities: &Array1<f64>) -> Result<CategoricalFeature1> {
        Ok(CategoricalFeature1 {
            probas_dirty: Array1::<f64>::zeros(probabilities.dim()),
            log_probas: probabilities.normalize_distribution()?.mapv(|x| x.log2()),
        })
    }
    pub fn dim(&self) -> usize {
        self.log_probas.dim()
    }
    pub fn check(&self) {
        if self.log_probas.iter().any(|&x| x > 0.) {
            panic!("Probabilities larger than one !");
        }
    }
}

// One-dimensional categorical distribution, given one external parameter
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pyclass)]
pub struct CategoricalFeature1g1 {
    pub log_probas: Array2<f64>,
    pub probas_dirty: Array2<f64>,
}

impl Feature<(usize, usize)> for CategoricalFeature1g1 {
    fn dirty_update(&mut self, observation: (usize, usize), likelihood: f64) {
        self.probas_dirty[[observation.0, observation.1]] += likelihood;
    }
    fn log_likelihood(&self, observation: (usize, usize)) -> f64 {
        self.log_probas[[observation.0, observation.1]]
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
            .log_probas
            .mapv(|x| x.exp2());
        for feat in iter {
            average_proba += &feat.log_probas.mapv(|x| x.exp2());
            len += 1;
        }
        CategoricalFeature1g1::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature1g1 {
    pub fn new(probabilities: &Array2<f64>) -> Result<CategoricalFeature1g1> {
        Ok(CategoricalFeature1g1 {
            probas_dirty: Array2::<f64>::zeros(probabilities.dim()),
            log_probas: probabilities.normalize_distribution()?.mapv(|x| x.log2()),
        })
    }
    pub fn dim(&self) -> (usize, usize) {
        self.log_probas.dim()
    }
    pub fn check(&self) {
        if self.log_probas.iter().any(|&x| x > 0.) {
            panic!("Probabilities larger than one !");
        }
    }
}

// Two-dimensional categorical distribution
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pyclass)]
pub struct CategoricalFeature2 {
    pub log_probas: Array2<f64>,
    pub probas_dirty: Array2<f64>,
}

impl Feature<(usize, usize)> for CategoricalFeature2 {
    fn dirty_update(&mut self, observation: (usize, usize), likelihood: f64) {
        self.probas_dirty[[observation.0, observation.1]] += likelihood;
    }
    fn log_likelihood(&self, observation: (usize, usize)) -> f64 {
        self.log_probas[[observation.0, observation.1]]
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
            .log_probas
            .mapv(|x| x.exp2());
        for feat in iter {
            average_proba = average_proba + feat.log_probas.mapv(|x| x.exp2());
            len += 1;
        }
        CategoricalFeature2::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature2 {
    pub fn new(probabilities: &Array2<f64>) -> Result<CategoricalFeature2> {
        Ok(CategoricalFeature2 {
            log_probas: probabilities
                .normalize_distribution_double()?
                .mapv(|x| x.log2()),
            probas_dirty: Array2::<f64>::zeros(probabilities.dim()),
        })
    }
    pub fn dim(&self) -> (usize, usize) {
        self.log_probas.dim()
    }
    pub fn check(&self) {
        if self.log_probas.iter().any(|&x| x > 0.) {
            panic!("Probabilities larger than one !");
        }
    }
}

// Two-dimensional categorical distribution, given one external parameter
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pyclass)]
pub struct CategoricalFeature2g1 {
    pub log_probas: Array3<f64>,
    pub probas_dirty: Array3<f64>,
}

impl Feature<(usize, usize, usize)> for CategoricalFeature2g1 {
    fn dirty_update(&mut self, observation: (usize, usize, usize), likelihood: f64) {
        self.probas_dirty[[observation.0, observation.1, observation.2]] += likelihood;
    }
    fn log_likelihood(&self, observation: (usize, usize, usize)) -> f64 {
        self.log_probas[[observation.0, observation.1, observation.2]]
    }
    fn cleanup(&self) -> Result<CategoricalFeature2g1> {
        let m = CategoricalFeature2g1::new(&self.probas_dirty)?;
        Ok(m)
    }
    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature2g1> + ExactSizeIterator,
    ) -> Result<CategoricalFeature2g1> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .log_probas
            .mapv(|x| x.exp2());
        for feat in iter {
            average_proba = average_proba + feat.log_probas.mapv(|x| x.exp2());
            len += 1;
        }
        CategoricalFeature2g1::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature2g1 {
    pub fn new(probabilities: &Array3<f64>) -> Result<CategoricalFeature2g1> {
        Ok(CategoricalFeature2g1 {
            probas_dirty: Array3::<f64>::zeros(probabilities.dim()),
            log_probas: probabilities
                .normalize_distribution_double()?
                .mapv(|x| x.log2()),
        })
    }
    pub fn dim(&self) -> (usize, usize, usize) {
        self.log_probas.dim()
    }
    pub fn check(&self) {
        if self.log_probas.iter().any(|&x| x > 0.) {
            panic!("Probabilities larger than one !");
        }
    }
}

// Most basic error model
#[derive(Default, Clone, Debug)]
#[cfg_attr(
    all(feature = "py_binds", feature = "py_o3"),
    pyclass(get_all, set_all)
)]
pub struct ErrorSingleNucleotide {
    pub error_rate: f64,
    logrs3: f64,
    log1mr: f64,
    // useful for dirty updating
    total_lengths_dirty: f64, // For each sequence, this saves Σ P(E) N_{err}(S(E))
    total_errors_dirty: f64,  // For each sequence, this saves Σ P(E) L(S(E))
}

impl ErrorSingleNucleotide {
    pub fn new(error_rate: f64) -> Result<ErrorSingleNucleotide> {
        if (error_rate < 0.)
            || (error_rate >= 1.)
            || (error_rate.is_nan())
            || (error_rate.is_infinite())
        {
            return Err(anyhow!(
                "Error in ErrorSingleNucleotide Feature creation. Negative/NaN/infinite error rate."
            ));
        }
        Ok(ErrorSingleNucleotide {
            error_rate,
            logrs3: (error_rate / 3.).log2(),
            log1mr: (1. - error_rate).log2(),
            total_lengths_dirty: 0.,
            total_errors_dirty: 0.,
        })
    }
}

impl Feature<(usize, usize)> for ErrorSingleNucleotide {
    /// Arguments
    /// - observation: "(nb of error, length of the sequence without insertion)"
    /// - likelihood: measured likelihood of the event
    fn dirty_update(&mut self, observation: (usize, usize), likelihood: f64) {
        self.total_lengths_dirty += likelihood * (observation.1 as f64);
        self.total_errors_dirty += likelihood * (observation.0 as f64);
    }

    /// Arguments
    /// - observation: "(nb of error, length of the sequence without insertion)"
    fn log_likelihood(&self, observation: (usize, usize)) -> f64 {
        (observation.0 as f64) * (self.logrs3)
            + ((observation.1 - observation.0) as f64) * self.log1mr
    }
    fn cleanup(&self) -> Result<ErrorSingleNucleotide> {
        // estimate the error_rate of the sequence from the dirty
        // estimate.
        let error_rate = if self.total_lengths_dirty == 0. {
            return ErrorSingleNucleotide::new(0.);
        } else {
            self.total_errors_dirty / self.total_lengths_dirty
        };

        Ok(ErrorSingleNucleotide {
            error_rate,
            logrs3: (error_rate / 3.).log2(),
            log1mr: (1. - error_rate).log2(),
            total_lengths_dirty: self.total_lengths_dirty,
            total_errors_dirty: self.total_errors_dirty,
        })
    }
    fn average(
        mut iter: impl Iterator<Item = ErrorSingleNucleotide> + ExactSizeIterator,
    ) -> Result<ErrorSingleNucleotide> {
        let first_feat = iter.next().ok_or(anyhow!("Cannot average empty vector"))?;
        let mut sum_err = first_feat.total_errors_dirty;
        let mut sum_length = first_feat.total_lengths_dirty;
        for feat in iter {
            sum_err += feat.total_errors_dirty;
            sum_length += feat.total_lengths_dirty;
        }
        println!("{} {}", sum_err, sum_length);
        ErrorSingleNucleotide::new(sum_err / sum_length)
    }
}

// Markov chain structure for Dna insertion
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pyclass)]
pub struct InsertionFeature {
    pub length_distribution: Array1<f64>,
    pub initial_distribution: Array1<f64>,
    pub transition_matrix: Array2<f64>,

    // log-transformed & include non-standard nucleotides
    log_initial_distribution_internal: Array1<f64>,
    log_transition_matrix_internal: Array2<f64>,
    log_length_distribution_internal: Array1<f64>,

    // for updating
    transition_matrix_dirty: Array2<f64>,
    length_distribution_dirty: Array1<f64>,
    initial_distribution_dirty: Array1<f64>,
}

impl Feature<&Dna> for InsertionFeature {
    fn dirty_update(&mut self, observation: &Dna, likelihood: f64) {
        if observation.is_empty() {
            self.length_distribution_dirty[0] += likelihood;
            return;
        }
        self.length_distribution_dirty[observation.len()] += likelihood;
        // N doesn't bring any information, so we ignore it in the update
        if observation.seq[0] != b'N' {
            self.initial_distribution_dirty[nucleotides_inv(observation.seq[0])] += likelihood;
        }
        for ii in 1..observation.len() {
            if (observation.seq[ii - 1] != b'N') && (observation.seq[ii] != b'N') {
                self.transition_matrix_dirty[[
                    nucleotides_inv(observation.seq[ii - 1]),
                    nucleotides_inv(observation.seq[ii]),
                ]] += likelihood / (observation.len() as f64 - 1.);
            }
        }
    }
    fn log_likelihood(&self, observation: &Dna) -> f64 {
        if observation.is_empty() {
            return self.log_length_distribution_internal[0];
        }
        let len = observation.len();
        let mut log_proba =
            self.log_initial_distribution_internal[nucleotides_inv(observation.seq[0])];
        for ii in 1..len {
            log_proba += self.log_transition_matrix_internal[[
                nucleotides_inv(observation.seq[ii - 1]),
                nucleotides_inv(observation.seq[ii]),
            ]];
        }
        log_proba + self.log_length_distribution_internal[len]
    }
    fn cleanup(&self) -> Result<InsertionFeature> {
        InsertionFeature::new(
            &self.length_distribution_dirty,
            &self.initial_distribution_dirty,
            &self.transition_matrix_dirty,
        )
    }
    fn average(
        mut iter: impl Iterator<Item = InsertionFeature> + ExactSizeIterator,
    ) -> Result<InsertionFeature> {
        let mut len = 1;
        let first_feat = iter.next().ok_or(anyhow!("Cannot average empty vector"))?;
        let mut average_init = first_feat.initial_distribution;
        let mut average_length = first_feat.length_distribution;
        let mut average_mat = first_feat.transition_matrix;
        for feat in iter {
            average_init = average_init + feat.initial_distribution;
            average_mat = average_mat + feat.transition_matrix;
            average_length = average_length + feat.length_distribution;
            len += 1;
        }
        InsertionFeature::new(
            &(average_length / (len as f64)),
            &(average_init / (len as f64)),
            &(average_mat / (len as f64)),
        )
    }
}

impl InsertionFeature {
    pub fn check(&self) {
        if self.log_transition_matrix_internal.iter().any(|&x| x > 0.) {
            panic!("Probabilities larger than one !");
        }
        if self
            .log_initial_distribution_internal
            .iter()
            .any(|&x| x > 0.)
        {
            panic!("Probabilities larger than one !");
        }
        if self
            .log_length_distribution_internal
            .iter()
            .any(|&x| x > 0.)
        {
            panic!("Probabilities larger than one !");
        }
    }

    pub fn new(
        length_distribution: &Array1<f64>,
        initial_distribution: &Array1<f64>,
        transition_matrix: &Array2<f64>,
    ) -> Result<InsertionFeature> {
        let mut m = InsertionFeature {
            length_distribution: length_distribution.normalize_distribution()?,
            transition_matrix: normalize_transition_matrix(transition_matrix)?,
            initial_distribution: initial_distribution.normalize_distribution()?,
            transition_matrix_dirty: Array2::<f64>::zeros(transition_matrix.dim()),
            initial_distribution_dirty: Array1::<f64>::zeros(initial_distribution.dim()),
            length_distribution_dirty: Array1::<f64>::zeros(length_distribution.dim()),
            log_transition_matrix_internal: Array2::<f64>::zeros((5, 5)),
            log_initial_distribution_internal: Array1::<f64>::zeros(5),
            log_length_distribution_internal: Array1::<f64>::zeros(length_distribution.dim()),
        };

        m.define_internal();
        Ok(m)
    }

    /// Just return the log likelihood from the length (faster)
    pub fn log_likelihood_length(&self, observation: usize) -> f64 {
        if observation >= self.log_length_distribution_internal.len() {
            return f64::NEG_INFINITY;
        }
        self.log_length_distribution_internal[observation]
    }

    pub fn get_parameters(&self) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        (
            self.length_distribution.clone(),
            self.initial_distribution.clone(),
            self.transition_matrix.clone(),
        )
    }

    /// 1- compute the log of the distribution (improve speed)
    /// 2- deal with undefined (N) nucleotides
    fn define_internal(&mut self) {
        self.log_length_distribution_internal = self.length_distribution.mapv(|x| x.log2());

        for ii in 0..4 {
            self.log_initial_distribution_internal[ii] = self.initial_distribution[ii].log2();
            for jj in 0..4 {
                self.log_transition_matrix_internal[[ii, jj]] =
                    self.transition_matrix[[ii, jj]].log2();
            }
        }
        self.log_initial_distribution_internal[4] = 0.;
        for ii in 0..5 {
            self.log_transition_matrix_internal[[ii, 4]] = 0.;
            self.log_transition_matrix_internal[[4, ii]] = 0.;
        }
    }
}
