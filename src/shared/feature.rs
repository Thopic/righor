use crate::sequence::utils::{nucleotides_inv, Dna};
use crate::shared::utils::{normalize_transition_matrix, Normalize, Normalize2, Normalize3};
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Array3};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
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
// - The `new` function should normalize the distribution correctly
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
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct CategoricalFeature1 {
    pub probas: Array1<f64>,
    pub probas_dirty: Array1<f64>,
}

impl Feature<usize> for CategoricalFeature1 {
    fn dirty_update(&mut self, observation: usize, likelihood: f64) {
        self.probas_dirty[[observation]] += likelihood;
    }
    fn likelihood(&self, observation: usize) -> f64 {
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
            probas_dirty: Array1::<f64>::zeros(probabilities.dim()),
            probas: probabilities.normalize_distribution()?,
        })
    }
    pub fn dim(&self) -> usize {
        self.probas.dim()
    }

    pub fn normalize(&self) -> Result<Self> {
        Self::new(&self.probas)
    }

    pub fn check(&self) {
        if self.probas.iter().any(|&x| x > 1.) {
            panic!("Probabilities larger than one !");
        }
    }
}

// One-dimensional categorical distribution, given one external parameter
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct CategoricalFeature1g1 {
    pub probas: Array2<f64>,
    pub probas_dirty: Array2<f64>,
}

impl Feature<(usize, usize)> for CategoricalFeature1g1 {
    fn dirty_update(&mut self, observation: (usize, usize), likelihood: f64) {
        self.probas_dirty[[observation.0, observation.1]] += likelihood;
    }
    fn likelihood(&self, observation: (usize, usize)) -> f64 {
        self.probas[[observation.0, observation.1]]
    }
    fn cleanup(&self) -> Result<CategoricalFeature1g1> {
        Ok(CategoricalFeature1g1 {
            probas_dirty: Array2::<f64>::zeros(self.probas.dim()),
            probas: self.probas_dirty.normalize_distribution_double()?,
        })
        //        CategoricalFeature1g1::new(&self.probas_dirty)
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
            average_proba += &feat.probas;
            len += 1;
        }
        CategoricalFeature1g1::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature1g1 {
    pub fn new(probabilities: &Array2<f64>) -> Result<CategoricalFeature1g1> {
        Ok(CategoricalFeature1g1 {
            probas_dirty: Array2::<f64>::zeros(probabilities.dim()),
            probas: probabilities.normalize_distribution()?,
        })
    }
    pub fn dim(&self) -> (usize, usize) {
        self.probas.dim()
    }
    pub fn normalize(&self) -> Result<Self> {
        Self::new(&self.probas)
    }

    pub fn check(&self) {
        if self.probas.iter().any(|&x| x > 1.) {
            panic!("Probabilities larger than one !");
        }
    }
}

// One-dimensional categorical distribution, given two external parameter
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct CategoricalFeature1g2 {
    pub probas: Array3<f64>,
    pub probas_dirty: Array3<f64>,
}

impl Feature<(usize, usize, usize)> for CategoricalFeature1g2 {
    fn dirty_update(&mut self, observation: (usize, usize, usize), likelihood: f64) {
        self.probas_dirty[[observation.0, observation.1, observation.2]] += likelihood;
    }
    fn likelihood(&self, observation: (usize, usize, usize)) -> f64 {
        self.probas[[observation.0, observation.1, observation.2]]
    }
    fn cleanup(&self) -> Result<CategoricalFeature1g2> {
        Ok(CategoricalFeature1g2 {
            probas_dirty: Array3::<f64>::zeros(self.probas.dim()),
            probas: self.probas_dirty.normalize_distribution_3()?,
        })
        //        CategoricalFeature1g2::new(&self.probas_dirty)
    }

    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature1g2> + ExactSizeIterator,
    ) -> Result<CategoricalFeature1g2> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas;
        for feat in iter {
            average_proba += &feat.probas;
            len += 1;
        }
        CategoricalFeature1g2::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature1g2 {
    pub fn new(probabilities: &Array3<f64>) -> Result<CategoricalFeature1g2> {
        Ok(CategoricalFeature1g2 {
            probas_dirty: Array3::<f64>::zeros(probabilities.dim()),
            probas: probabilities.normalize_distribution()?,
        })
    }
    pub fn dim(&self) -> (usize, usize, usize) {
        self.probas.dim()
    }
    pub fn normalize(&self) -> Result<Self> {
        Self::new(&self.probas)
    }

    pub fn check(&self) {
        if self.probas.iter().any(|&x| x > 1.) {
            panic!("Probabilities larger than one !");
        }
    }
}

// Two-dimensional categorical distribution
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct CategoricalFeature2 {
    pub probas: Array2<f64>,
    pub probas_dirty: Array2<f64>,
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
            probas: probabilities.normalize_distribution_double()?,
            probas_dirty: Array2::<f64>::zeros(probabilities.dim()),
        })
    }
    pub fn dim(&self) -> (usize, usize) {
        self.probas.dim()
    }
    pub fn normalize(&self) -> Result<Self> {
        Self::new(&self.probas)
    }

    pub fn check(&self) {
        if self.probas.iter().any(|&x| x > 1.) {
            panic!("Probabilities larger than one !");
        }
    }
}

// Two-dimensional categorical distribution, given one external parameter
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct CategoricalFeature2g1 {
    pub probas: Array3<f64>,
    pub probas_dirty: Array3<f64>,
}

impl Feature<(usize, usize, usize)> for CategoricalFeature2g1 {
    fn dirty_update(&mut self, observation: (usize, usize, usize), likelihood: f64) {
        self.probas_dirty[[observation.0, observation.1, observation.2]] += likelihood;
    }
    fn likelihood(&self, observation: (usize, usize, usize)) -> f64 {
        self.probas[[observation.0, observation.1, observation.2]]
    }
    fn cleanup(&self) -> Result<CategoricalFeature2g1> {
        Ok(CategoricalFeature2g1 {
            probas_dirty: Array3::<f64>::zeros(self.probas.dim()),
            probas: self.probas_dirty.normalize_distribution_3()?,
        })
        // let m = CategoricalFeature2g1::new(&self.probas_dirty)?;
        // Ok(m)
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
            probas_dirty: Array3::<f64>::zeros(probabilities.dim()),
            probas: probabilities.normalize_distribution_double()?,
        })
    }
    pub fn dim(&self) -> (usize, usize, usize) {
        self.probas.dim()
    }
    pub fn normalize(&self) -> Result<Self> {
        Self::new(&self.probas)
    }

    pub fn check(&self) {
        if self.probas.iter().any(|&x| x > 1.) {
            panic!("Probabilities larger than one !");
        }
    }
}

// Three-dimensional distribution
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct CategoricalFeature3 {
    pub probas: Array3<f64>,
    pub probas_dirty: Array3<f64>,
}

impl Feature<(usize, usize, usize)> for CategoricalFeature3 {
    fn dirty_update(&mut self, observation: (usize, usize, usize), likelihood: f64) {
        self.probas_dirty[[observation.0, observation.1, observation.2]] += likelihood;
    }
    fn likelihood(&self, observation: (usize, usize, usize)) -> f64 {
        self.probas[[observation.0, observation.1, observation.2]]
    }
    fn cleanup(&self) -> Result<CategoricalFeature3> {
        let m = CategoricalFeature3::new(&self.probas_dirty)?;
        Ok(m)
    }
    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature3> + ExactSizeIterator,
    ) -> Result<CategoricalFeature3> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas;
        for feat in iter {
            average_proba = average_proba + feat.probas;
            len += 1;
        }
        CategoricalFeature3::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature3 {
    pub fn new(probabilities: &Array3<f64>) -> Result<CategoricalFeature3> {
        Ok(CategoricalFeature3 {
            probas_dirty: Array3::<f64>::zeros(probabilities.dim()),
            probas: probabilities.normalize_distribution_3()?,
        })
    }
    pub fn dim(&self) -> (usize, usize, usize) {
        self.probas.dim()
    }
    pub fn normalize(&self) -> Result<Self> {
        Self::new(&self.probas)
    }

    pub fn check(&self) {
        if self.probas.iter().any(|&x| x > 1.) {
            panic!("Probabilities larger than one !");
        }
    }
}

// Most basic error model
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
pub struct ErrorSingleNucleotide {
    pub error_rate: f64,
    logrs3: f64,
    log1mr: f64,
    total_lengths: f64, // For each sequence, this saves Σ P(E) L(S(E))
    total_errors: f64,  // For each sequence, this saves Σ P(E) N_{err}(S(E))
    // useful for dirty updating
    total_lengths_dirty: f64,
    total_errors_dirty: f64,
    total_probas_dirty: f64, // For each sequence, this saves Σ P(E)
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
            total_probas_dirty: 0.,
            ..Default::default()
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
        self.total_probas_dirty += likelihood;
    }

    /// Arguments
    /// - observation: "(nb of error, length of the sequence without insertion)"
    /// The complete formula is likelihood = (r/3)^(nb error) * (1-r)^(length - nb error)
    fn likelihood(&self, observation: (usize, usize)) -> f64 {
        if observation.0 == 0 {
            return (observation.1 as f64 * self.log1mr).exp2();
        }
        ((observation.0 as f64) * self.logrs3
            + ((observation.1 - observation.0) as f64) * self.log1mr)
            .exp2()
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
            total_lengths: self.total_lengths_dirty / self.total_probas_dirty,
            total_errors: self.total_errors_dirty / self.total_probas_dirty,
            total_probas_dirty: 0.,
            total_lengths_dirty: 0.,
            total_errors_dirty: 0.,
        })
    }
    fn average(
        mut iter: impl Iterator<Item = ErrorSingleNucleotide> + ExactSizeIterator,
    ) -> Result<ErrorSingleNucleotide> {
        let first_feat = iter.next().ok_or(anyhow!("Cannot average empty vector"))?;
        let mut sum_err = first_feat.total_errors;
        let mut sum_length = first_feat.total_lengths;
        for feat in iter {
            sum_err += feat.total_errors;
            sum_length += feat.total_lengths;
        }
        if sum_length == 0. {
            return ErrorSingleNucleotide::new(0.);
        }
        ErrorSingleNucleotide::new(sum_err / sum_length)
    }
}

// Markov chain structure for Dna insertion
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct InsertionFeature {
    pub length_distribution: Array1<f64>,
    //pub initial_distribution: Array1<f64>, // This should not be here anymore, rm
    pub transition_matrix: Array2<f64>,

    // include non-standard nucleotides
    transition_matrix_internal: Array2<f64>,

    // for updating
    transition_matrix_dirty: Array2<f64>,
    length_distribution_dirty: Array1<f64>,
    //  initial_distribution_dirty: Array1<f64>,
}

impl Feature<&Dna> for InsertionFeature {
    /// Observation plus one contains the sequence of interest with the nucleotide
    /// preceding it, so if we're interested in the insertion CTGGC that pops up
    /// in the sequence CAACTGGCAC we would send ACTGGC.
    fn dirty_update(&mut self, observation_plus_one: &Dna, likelihood: f64) {
        if observation_plus_one.len() == 1 {
            self.length_distribution_dirty[0] += likelihood;
            return;
        }
        self.length_distribution_dirty[observation_plus_one.len() - 1] += likelihood;

        for ii in 1..observation_plus_one.len() {
            // TODO: The way I deal with N is not quite exact, need to fix that (not a big deal though)
            // if (likelihood != 0.) {
            //     println!("{}\t{}", observation_plus_one.get_string(), likelihood);
            // }
            if (observation_plus_one.seq[ii - 1] != b'N') && (observation_plus_one.seq[ii] != b'N')
            {
                self.transition_matrix_dirty[[
                    nucleotides_inv(observation_plus_one.seq[ii - 1]),
                    nucleotides_inv(observation_plus_one.seq[ii]),
                ]] += likelihood
            }
        }
    }

    /// Observation plus one contains the sequence of interest with the nucleotide
    /// preceding it, so if we're interested in the insertion CTGGC that pops up
    /// in the sequence CAACTGGCAC we would send ACTGGC.
    fn likelihood(&self, observation_plus_one: &Dna) -> f64 {
        if observation_plus_one.len() > self.length_distribution.len() {
            return 0.;
        }
        if observation_plus_one.len() == 1 {
            return self.length_distribution[0];
        }
        let len = observation_plus_one.len() - 1;
        let mut proba = 1.;
        for ii in 1..len + 1 {
            proba *= self.transition_matrix_internal[[
                nucleotides_inv(observation_plus_one.seq[ii - 1]),
                nucleotides_inv(observation_plus_one.seq[ii]),
            ]];
        }
        // println!(
        //     "likelihood: {}\t{}",
        //     observation_plus_one.get_string(),
        //     proba * self.length_distribution[len]
        // );
        proba * self.length_distribution[len]
    }
    fn cleanup(&self) -> Result<InsertionFeature> {
        let mut m = InsertionFeature {
            length_distribution: self.length_distribution_dirty.normalize_distribution()?,
            // we shouldn't normalize the transition matrix per line, because bias on
            // the auto-transition X -> X
            transition_matrix: self
                .transition_matrix_dirty
                .normalize_distribution_double()?,
            transition_matrix_dirty: Array2::<f64>::zeros(self.transition_matrix.dim()),
            length_distribution_dirty: Array1::<f64>::zeros(self.length_distribution.dim()),
            transition_matrix_internal: Array2::<f64>::zeros((5, 5)),
        };

        m.define_internal();
        Ok(m)
        // println!("{:?}", self.transition_matrix_dirty);

        // InsertionFeature::new(
        //     &self.length_distribution_dirty,
        //     &self.transition_matrix_dirty,
        // )
    }
    fn average(
        mut iter: impl Iterator<Item = InsertionFeature> + ExactSizeIterator,
    ) -> Result<InsertionFeature> {
        let mut len = 1;
        let first_feat = iter.next().ok_or(anyhow!("Cannot average empty vector"))?;
        let mut average_length = first_feat.length_distribution;
        let mut average_mat = first_feat.transition_matrix;
        for feat in iter {
            average_mat = average_mat + feat.transition_matrix;
            average_length = average_length + feat.length_distribution;
            len += 1;
        }
        InsertionFeature::new(
            &(average_length / (len as f64)),
            &(average_mat / (len as f64)),
        )
    }
}

impl InsertionFeature {
    pub fn check(&self) {
        if self.transition_matrix_internal.iter().any(|&x| x > 1.) {
            panic!("Probabilities larger than one !");
        }
        if self.length_distribution.iter().any(|&x| x > 1.) {
            panic!("Probabilities larger than one !");
        }
    }

    pub fn new(
        length_distribution: &Array1<f64>,
        transition_matrix: &Array2<f64>,
    ) -> Result<InsertionFeature> {
        let mut m = InsertionFeature {
            length_distribution: length_distribution.normalize_distribution()?,
            transition_matrix: normalize_transition_matrix(transition_matrix)?,
            transition_matrix_dirty: Array2::<f64>::zeros(transition_matrix.dim()),
            length_distribution_dirty: Array1::<f64>::zeros(length_distribution.dim()),
            transition_matrix_internal: Array2::<f64>::zeros((5, 5)),
        };

        m.define_internal();
        Ok(m)
    }

    pub fn normalize(&self) -> Result<Self> {
        Self::new(&self.length_distribution, &self.transition_matrix)
    }

    pub fn get_parameters(&self) -> (Array1<f64>, Array2<f64>) {
        (
            self.length_distribution.clone(),
            self.transition_matrix.clone(),
        )
    }

    pub fn max_nb_insertions(&self) -> usize {
        self.length_distribution.len()
    }

    /// deal with undefined (N) nucleotides
    fn define_internal(&mut self) {
        for ii in 0..4 {
            for jj in 0..4 {
                self.transition_matrix_internal[[ii, jj]] = self.transition_matrix[[ii, jj]];
            }
        }
        for ii in 0..5 {
            self.transition_matrix_internal[[ii, 4]] = 0.;
            self.transition_matrix_internal[[4, ii]] = 0.;
        }
    }
}
