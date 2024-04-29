use crate::shared::utils::{normalize_transition_matrix, Normalize, Normalize2, Normalize3};
use crate::shared::{nucleotides_inv, Dna, InferenceParameters};
use crate::{v_dj, vdj};
use anyhow::{anyhow, Result};
use dyn_clone::DynClone;
use ndarray::{Array1, Array2, Array3, Axis};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

// This class define different type of Feature
// Feature are used during the expectation maximization process
// In short you need:
// - a function computing the likelihood of the feature `likelihood`
// - a function that allows to update the probability distribution
//   when new observations are made.
//   This update is done lazily, for speed reason we don't want to
//   redefine the function everytime.
// - scale_dirty apply a general scaling to everything
// - The `new` function should normalize the distribution correctly
// This is the general idea, there's quite a lot of boiler plate
// code for the different type of categorical features (categorical features in
// 1d, in 1d but given another parameter, in 2d ...)

pub trait Feature<T> {
    fn dirty_update(&mut self, observation: T, likelihood: f64);
    fn likelihood(&self, observation: T) -> f64;
    fn scale_dirty(&mut self, factor: f64);
    fn average(iter: impl Iterator<Item = Self> + ExactSizeIterator + Clone) -> Result<Self>
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

    fn scale_dirty(&mut self, factor: f64) {
        self.probas_dirty *= factor;
    }
    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature1> + ExactSizeIterator + Clone,
    ) -> Result<CategoricalFeature1> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas_dirty;
        for feat in iter {
            average_proba = average_proba + feat.probas_dirty;
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
    fn scale_dirty(&mut self, factor: f64) {
        self.probas_dirty *= factor;
    }
    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature1g1> + ExactSizeIterator + Clone,
    ) -> Result<CategoricalFeature1g1> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas_dirty;
        for feat in iter {
            average_proba += &feat.probas_dirty;
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

    fn scale_dirty(&mut self, factor: f64) {
        self.probas_dirty *= factor;
    }

    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature1g2> + ExactSizeIterator + Clone,
    ) -> Result<CategoricalFeature1g2> {
        let mut len = 1;

        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas_dirty;
        for feat in iter {
            average_proba += &feat.probas_dirty;
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

    fn scale_dirty(&mut self, factor: f64) {
        self.probas_dirty *= factor;
    }
    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature2> + ExactSizeIterator + Clone,
    ) -> Result<CategoricalFeature2> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas_dirty;
        for feat in iter {
            average_proba += &feat.probas_dirty;
            len += 1;
        }
        CategoricalFeature2::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature2 {
    pub fn new(probabilities: &Array2<f64>) -> Result<CategoricalFeature2> {
        let probas = probabilities.normalize_distribution_double()?;

        Ok(CategoricalFeature2 {
            probas,
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

    fn scale_dirty(&mut self, factor: f64) {
        self.probas_dirty *= factor;
    }
    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature2g1> + ExactSizeIterator + Clone,
    ) -> Result<CategoricalFeature2g1> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas_dirty;
        for feat in iter {
            average_proba = average_proba + feat.probas_dirty;
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

    fn scale_dirty(&mut self, factor: f64) {
        self.probas_dirty *= factor;
    }

    fn average(
        mut iter: impl Iterator<Item = CategoricalFeature3> + ExactSizeIterator + Clone,
    ) -> Result<CategoricalFeature3> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas_dirty;
        for feat in iter {
            average_proba = average_proba + feat.probas_dirty;
            len += 1;
        }
        CategoricalFeature3::new(&(average_proba / (len as f64)))
    }
}

impl CategoricalFeature3 {
    pub fn new(probabilities: &Array3<f64>) -> Result<CategoricalFeature3> {
        let probas = probabilities.normalize_distribution_3()?;

        Ok(CategoricalFeature3 {
            probas_dirty: Array3::<f64>::zeros(probabilities.dim()),
            probas,
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
    // total_lengths: f64, // For each sequence, this saves Σ P(E) L(S(E))
    // total_errors: f64,  // For each sequence, this saves Σ P(E) N_{err}(S(E))
    // useful for dirty updating
    total_lengths_dirty: f64,
    total_errors_dirty: f64,
    total_probas_dirty: f64, // For each sequence, this saves Σ P(E)
}

impl ErrorSingleNucleotide {
    pub fn new(error_rate: f64) -> Result<ErrorSingleNucleotide> {
        if !(0. ..1.).contains(&error_rate) || (error_rate.is_nan()) || (error_rate.is_infinite()) {
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

    fn scale_dirty(&mut self, factor: f64) {
        self.total_errors_dirty *= factor;
        self.total_lengths_dirty *= factor;
    }

    // fn cleanup(&self) -> Result<ErrorSingleNucleotide> {
    //     // estimate the error_rate of the sequence from the dirty
    //     // estimate.
    //     let error_rate = if self.total_lengths_dirty == 0. {
    //         return ErrorSingleNucleotide::new(0.);
    //     } else {
    //         self.total_errors_dirty / self.total_lengths_dirty
    //     };

    //     Ok(ErrorSingleNucleotide {
    //         error_rate,
    //         logrs3: (error_rate / 3.).log2(),
    //         log1mr: (1. - error_rate).log2(),
    //         total_lengths: self.total_lengths_dirty / self.total_probas_dirty,
    //         total_errors: self.total_errors_dirty / self.total_probas_dirty,
    //         total_probas_dirty: 0.,
    //         total_lengths_dirty: 0.,
    //         total_errors_dirty: 0.,
    //     })
    // }
    fn average(
        mut iter: impl Iterator<Item = ErrorSingleNucleotide> + ExactSizeIterator + Clone,
    ) -> Result<ErrorSingleNucleotide> {
        let first_feat = iter.next().ok_or(anyhow!("Cannot average empty vector"))?;
        let mut sum_err = first_feat.total_errors_dirty;
        let mut sum_length = first_feat.total_lengths_dirty;
        for feat in iter {
            sum_err += feat.total_errors_dirty;
            sum_length += feat.total_lengths_dirty;
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
    pub transition_matrix_dirty: Array2<f64>,
    pub length_distribution_dirty: Array1<f64>,
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

    fn scale_dirty(&mut self, factor: f64) {
        self.length_distribution_dirty *= factor;
        self.transition_matrix_dirty *= factor;
    }

    fn average(
        mut iter: impl Iterator<Item = InsertionFeature> + ExactSizeIterator + Clone,
    ) -> Result<InsertionFeature> {
        let mut len = 1;
        let first_feat = iter.next().ok_or(anyhow!("Cannot average empty vector"))?;
        let mut average_length = first_feat.length_distribution_dirty;
        let mut average_mat = first_feat.transition_matrix_dirty;
        for feat in iter {
            average_mat = average_mat + feat.transition_matrix_dirty;
            average_length = average_length + feat.length_distribution_dirty;
            len += 1;
        }
        InsertionFeature::new(
            &(average_length / (len as f64)),
            &(average_mat / (len as f64)),
        )
    }
}

impl InsertionFeature {
    pub fn correct_for_uniform_error_rate(&mut self, r: f64) {
        let matrix = 1. / (1. - r) * (Array2::eye(4) + r / 4. * Array2::ones((4, 4)));
        self.transition_matrix = matrix.dot(&self.transition_matrix.dot(&matrix));
    }

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
            self.transition_matrix_internal[[ii, 4]] = 1.;
            self.transition_matrix_internal[[4, ii]] =
                self.transition_matrix.sum_axis(Axis(0))[[ii]];
        }
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug, PartialEq)]
pub struct InfEvent {
    pub v_index: usize,
    pub v_start_gene: usize, // start of the sequence in the V gene
    pub j_index: usize,
    pub j_start_seq: usize, // start of the palindromic J gene (with all dels) in the sequence
    pub d_index: usize,
    // position of the v,d,j genes in the sequence
    pub end_v: i64,
    pub start_d: i64,
    pub end_d: i64,
    pub start_j: i64,

    // sequences (only added after the inference is over)
    pub ins_vd: Option<Dna>,
    pub ins_dj: Option<Dna>,
    pub d_segment: Option<Dna>,
    pub sequence: Option<Dna>,
    pub cdr3: Option<Dna>,
    pub full_sequence: Option<Dna>,
    pub reconstructed_sequence: Option<Dna>,

    // likelihood (pgen + perror)
    pub likelihood: f64,
}

#[derive(Clone, Debug)]
pub enum FeaturesGeneric {
    VDJ(vdj::Features),
    VxDJ(v_dj::Features),
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
#[derive(Clone)]
pub struct ResultInference {
    pub likelihood: f64,
    pub pgen: f64,
    pub best_event: Option<InfEvent>,
    pub best_likelihood: f64,
    pub features: Option<Box<dyn FeaturesTrait>>,
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl ResultInference {
    #[getter]
    pub fn get_likelihood(&self) -> f64 {
        self.likelihood
    }
    #[getter]
    pub fn get_pgen(&self) -> f64 {
        self.pgen
    }
    #[getter]
    #[pyo3(name = "best_event")]
    pub fn py_get_best_event(&self) -> Option<InfEvent> {
        self.get_best_event()
    }
    #[getter]
    pub fn get_likelihood_best_event(&self) -> f64 {
        self.best_likelihood
    }
}

/// A Result class that's easily readable
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all))]
pub struct ResultHuman {
    pub n_cdr3: String,
    pub aa_cdr3: String,
    pub likelihood: f64,
    pub pgen: f64,
    pub likelihood_ratio_best: f64,
    pub seq: String,
    pub full_seq: String,
    pub reconstructed_seq: String,
    pub aligned_v: String,
    pub aligned_j: String,
    pub v_name: String,
    pub j_name: String,
}

impl ResultInference {
    pub fn display(&self, model: &vdj::Model) -> Result<String> {
        if self.best_event.is_none() {
            return Ok(format!(
                "Result:\n\
		 - Likelihood: {}\n\
		 - Pgen: {}\n",
                self.likelihood, self.pgen
            ));
        }

        let rh = self.to_human(model)?;
        Ok(format!(
            "Result:\n\
	     \tLikelihood: {:.2e}, pgen: {:.2e}\n\
	     \tMost likely event:\n\
	     \t- CDR3 (nucleotides): {} \n\
	     \t- CDR3 (amino acids): {} \n\
	     \t- V name: {} \n\
	     \t- J name: {} \n\
	     \t- likelihood ratio: {} \n ",
            self.likelihood,
            self.pgen,
            rh.n_cdr3,
            rh.aa_cdr3,
            rh.v_name,
            rh.j_name,
            rh.likelihood_ratio_best
        ))
    }

    /// Translate the result to an easier to read/print version
    pub fn to_human(&self, model: &vdj::Model) -> Result<ResultHuman> {
        let best_event = self.get_best_event().ok_or(anyhow!("No event"))?;

        let translated_cdr3 = if best_event.cdr3.clone().unwrap().len() % 3 == 0 {
            best_event
                .cdr3
                .clone()
                .unwrap()
                .translate()
                .unwrap()
                .to_string()
        } else {
            String::new()
        };

        let reconstructed_seq = best_event
            .reconstructed_sequence
            .clone()
            .unwrap()
            .get_string();
        let width = reconstructed_seq.len();

        let aligned_v = format!(
            "{:width$}",
            model.seg_vs[best_event.v_index].seq.get_string(),
            width = width
        );
        let aligned_j = format!(
            "{:>width$}",
            model.seg_js[best_event.j_index].seq.get_string(),
            width = width
        );

        Ok(ResultHuman {
            n_cdr3: best_event.cdr3.clone().unwrap().get_string(),
            aa_cdr3: translated_cdr3,
            likelihood: self.likelihood,
            pgen: self.pgen,
            likelihood_ratio_best: best_event.likelihood / self.likelihood,
            seq: best_event.sequence.clone().unwrap().get_string(),
            full_seq: best_event.full_sequence.clone().unwrap().get_string(),
            reconstructed_seq,
            aligned_v,
            aligned_j,
            v_name: model.get_v_gene(&best_event),
            j_name: model.get_j_gene(&best_event),
        })
    }

    pub fn impossible() -> ResultInference {
        ResultInference {
            likelihood: 0.,
            pgen: 0.,
            best_event: None,
            best_likelihood: 0.,
            features: None,
        }
    }
    pub fn set_best_event(&mut self, ev: InfEvent, ip: &InferenceParameters) {
        if ip.store_best_event {
            self.best_event = Some(ev);
        }
    }
    pub fn get_best_event(&self) -> Option<InfEvent> {
        self.best_event.clone()
    }
    /// I just store the necessary stuff in the Event variable while looping
    /// Fill event add enough to be able to completely recreate the sequence
    pub fn fill_event(&mut self, model: &vdj::Model, sequence: &vdj::Sequence) -> Result<()> {
        if self.best_event.is_some() {
            let mut event = self.best_event.clone().unwrap();
            event.ins_vd = Some(
                sequence
                    .sequence
                    .extract_padded_subsequence(event.end_v, event.start_d),
            );
            event.ins_dj = Some(
                sequence
                    .sequence
                    .extract_padded_subsequence(event.end_d, event.start_j),
            );

            event.d_segment = Some(
                sequence
                    .sequence
                    .extract_padded_subsequence(event.start_d, event.end_d),
            );

            event.sequence = Some(sequence.sequence.clone());

            let cdr3_pos_v = model.seg_vs[event.v_index]
                .cdr3_pos
                .ok_or(anyhow!("Gene not loaded correctly"))?;
            let cdr3_pos_j = model.seg_js[event.j_index]
                .cdr3_pos
                .ok_or(anyhow!("Gene not loaded correctly"))?;

            let start_cdr3 = cdr3_pos_v as i64 - event.v_start_gene as i64;

            // careful, cdr3_pos_j does not! include the palindromic insertions
            // or the last nucleotide
            let end_cdr3 = event.j_start_seq as i64 + cdr3_pos_j as i64 - model.range_del_j.0 + 3;

            event.cdr3 = Some(
                sequence
                    .sequence
                    .extract_padded_subsequence(start_cdr3, end_cdr3),
            );

            let gene_v = model.seg_vs[event.v_index]
                .clone()
                .seq_with_pal
                .ok_or(anyhow!("Model not loaded correctly"))?;

            let gene_j = model.seg_js[event.j_index]
                .clone()
                .seq_with_pal
                .ok_or(anyhow!("Model not loaded correctly"))?;

            let mut full_seq = gene_v.extract_subsequence(0, event.v_start_gene);
            full_seq.extend(&sequence.sequence);
            full_seq.extend(
                &gene_j
                    .extract_subsequence(sequence.sequence.len() - event.j_start_seq, gene_j.len()),
            );
            event.full_sequence = Some(full_seq);

            let mut reconstructed_seq =
                gene_v.extract_subsequence(0, (event.end_v + event.v_start_gene as i64) as usize);
            reconstructed_seq.extend(&event.ins_vd.clone().unwrap());
            reconstructed_seq.extend(&event.d_segment.clone().unwrap());
            reconstructed_seq.extend(&event.ins_dj.clone().unwrap());
            reconstructed_seq.extend(&gene_j.extract_padded_subsequence(
                event.start_j - event.j_start_seq as i64,
                gene_j.len() as i64,
            ));
            event.reconstructed_sequence = Some(reconstructed_seq);
            self.best_event = Some(event);
        }
        Ok(())
    }
}

pub trait FeaturesTrait: Send + Sync + DynClone {
    fn delv(&self) -> &CategoricalFeature1g1;
    fn delj(&self) -> &CategoricalFeature1g1;
    fn deld(&self) -> &CategoricalFeature2g1;
    fn insvd(&self) -> &InsertionFeature;
    fn insdj(&self) -> &InsertionFeature;
    fn error(&self) -> &ErrorSingleNucleotide;
    fn delv_mut(&mut self) -> &mut CategoricalFeature1g1;
    fn delj_mut(&mut self) -> &mut CategoricalFeature1g1;
    fn deld_mut(&mut self) -> &mut CategoricalFeature2g1;
    fn insvd_mut(&mut self) -> &mut InsertionFeature;
    fn insdj_mut(&mut self) -> &mut InsertionFeature;
    fn error_mut(&mut self) -> &mut ErrorSingleNucleotide;
    fn generic(&self) -> FeaturesGeneric;
    fn new(model: &vdj::Model) -> Result<Self>
    where
        Self: Sized;
    fn infer(
        &mut self,
        sequence: &vdj::Sequence,
        ip: &InferenceParameters,
    ) -> Result<ResultInference>;
    fn update_model(&self, model: &mut vdj::Model) -> Result<()>;
    fn average(features: Vec<Self>) -> Result<Self>
    where
        Self: Sized;
}

dyn_clone::clone_trait_object!(FeaturesTrait);
