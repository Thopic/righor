use crate::shared::likelihood::Likelihood;
use crate::shared::sequence::Dna;
use crate::shared::utils::{Normalize, Normalize2, Normalize3};
use crate::shared::DNAMarkovChain;
use crate::shared::ModelStructure;
use crate::shared::{errors::FeatureError, DnaLike, InferenceParameters};
use crate::vdj::Model as ModelVDJ;
use crate::{v_dj, vdj};
use anyhow::{anyhow, Result};
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use std::fmt::Debug;

//use crate::shared::distributions::calc_steady_state_dist;

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
        assert!(
            !self.probas.iter().any(|&x| x > 1.),
            "Probabilities larger than one !"
        );
    }
    pub fn average(
        mut iter: impl Iterator<Item = CategoricalFeature1> + Clone,
    ) -> Result<Vec<CategoricalFeature1>> {
        let mut len = 1;
        let mut average_proba = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas_dirty;
        for feat in iter {
            average_proba = average_proba + feat.probas_dirty;
            len += 1;
        }
        let new_feat = CategoricalFeature1::new(&(average_proba / (len as f64)))?;
        Ok(vec![new_feat; len])
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
        assert!(
            !self.probas.iter().any(|&x| x > 1.),
            "Probabilities larger than one !"
        );
    }
    pub fn average(
        mut iter: impl Iterator<Item = CategoricalFeature1g1> + Clone,
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
        let average_feat = CategoricalFeature1g1::new(&(average_proba / (f64::from(len))))?;
        Ok(average_feat)
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
        assert!(
            !self.probas.iter().any(|&x| x > 1.),
            "Probabilities larger than one !"
        );
    }
    pub fn average(
        mut iter: impl Iterator<Item = CategoricalFeature1g2> + Clone,
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
        let average_feat = CategoricalFeature1g2::new(&(average_proba / (f64::from(len))))?;
        Ok(average_feat)
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
        assert!(
            self.probas.iter().any(|&x| x > 1.),
            "Probabilities larger than one !"
        );
    }
    pub fn average(
        mut iter: impl Iterator<Item = CategoricalFeature2> + Clone,
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
        let average_feat = CategoricalFeature2::new(&(average_proba / (f64::from(len))))?;
        Ok(average_feat)
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
        assert!(
            self.probas.iter().any(|&x| x > 1.),
            "Probabilities larger than one !"
        );
    }
    pub fn average(
        mut iter: impl Iterator<Item = CategoricalFeature2g1> + Clone,
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
        let average_feat = CategoricalFeature2g1::new(&(average_proba / (f64::from(len))))?;
        Ok(average_feat)
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
        assert!(
            !self.probas.iter().any(|&x| x > 1.),
            "Probabilities larger than one !"
        );
    }
    pub fn average(
        mut iter: impl Iterator<Item = CategoricalFeature3> + Clone,
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
        let average_feat = CategoricalFeature3::new(&(average_proba / (f64::from(len))))?;
        Ok(average_feat)
    }
}

// Markov chain structure for Dna insertion
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct InsertionFeature {
    pub length_distribution: Array1<f64>,

    // give the likelihood of a given sequence
    pub transition: Arc<DNAMarkovChain>,

    // for updating
    pub transition_matrix_dirty: Array2<f64>,
    pub length_distribution_dirty: Array1<f64>,
}

impl InsertionFeature {
    /// - observation: sequence of interest
    /// - `first_nucleotide`: the nucleotide preceding the sequence of interest
    /// - `likelihood`: likelihood value to be updated
    pub fn dirty_update(
        &mut self,
        observation: &DnaLike,
        first_nucleotide: usize,
        likelihood: f64,
    ) {
        if observation.is_empty() {
            self.length_distribution_dirty[0] += likelihood;
            return;
        }
        self.length_distribution_dirty[observation.len()] += likelihood;

        self.transition_matrix_dirty =
            self.transition
                .update(observation, first_nucleotide, likelihood);
    }

    /// - observation: sequence of interest
    /// - `first_nucleotide`: the nucleotide preceding the sequence of interest
    pub fn likelihood(&self, observation: &DnaLike, first_nucleotide: usize) -> Likelihood {
        if observation.len() >= self.length_distribution.len() {
            return Likelihood::zero(observation);
        }
        if observation.is_empty() {
            return Likelihood::identity(observation) * self.length_distribution[0];
        }
        let len = observation.len();
        self.transition.likelihood(observation, first_nucleotide) * self.length_distribution[len]
    }

    pub fn scale_dirty(&mut self, factor: f64) {
        self.length_distribution_dirty *= factor;
        self.transition_matrix_dirty *= factor;
    }
}

impl InsertionFeature {
    pub fn correct_for_error(&self, err: &FeatureError) -> InsertionFeature {
        // The error rate make the inferred value of the transition rate wrong
        // we correct it using the current error rate estimate.

        match err {
            FeatureError::ConstantRate(f) => {
                let rho = 4. * f.error_rate / 3.;
                let matrix = 1. / (1. - rho) * (Array2::eye(4) - rho / 4. * Array2::ones((4, 4)));
                let mut insfeat = self.clone();
                // we just modify the "dirty" matrix (no choice ...)
                insfeat.transition_matrix_dirty =
                    matrix.dot(&insfeat.transition_matrix_dirty.dot(&matrix));
                insfeat
            }
            FeatureError::UniformRate(f) => {
                let mut insfeat = self.clone();
                let rho = 4. * f.get_error_rate() / 3.;
                let matrix = 1. / (1. - rho) * (Array2::eye(4) - rho / 4. * Array2::ones((4, 4)));
                insfeat.transition_matrix_dirty =
                    matrix.dot(&insfeat.transition_matrix_dirty.dot(&matrix));
                insfeat
            }
        }
    }

    pub fn check(&self) {
        assert!(
            !self.transition.transition_matrix.iter().any(|&x| x > 1.),
            "Probabilities larger than one !"
        );
        assert!(
            !self.length_distribution.iter().any(|&x| x > 1.),
            "Probabilities larger than one !"
        );
    }

    pub fn new(
        length_distribution: &Array1<f64>,
        transition: Arc<DNAMarkovChain>,
    ) -> Result<InsertionFeature> {
        let dim = transition.transition_matrix.dim();
        let m = InsertionFeature {
            length_distribution: length_distribution.normalize_distribution()?,
            transition,
            transition_matrix_dirty: Array2::<f64>::zeros(dim),
            length_distribution_dirty: Array1::<f64>::zeros(length_distribution.dim()),
        };

        Ok(m)
    }

    pub fn normalize(&self) -> Result<Self> {
        Self::new(&self.length_distribution, self.transition.clone())
    }

    pub fn get_parameters(&self) -> (Array1<f64>, Array2<f64>) {
        (
            self.length_distribution.clone(),
            self.transition.transition_matrix.clone(),
        )
    }

    pub fn max_nb_insertions(&self) -> usize {
        self.length_distribution.len()
    }

    pub fn average(
        mut iter: impl Iterator<Item = InsertionFeature> + Clone,
    ) -> Result<InsertionFeature> {
        let mut len = 1;
        let first_feat = iter.next().ok_or(anyhow!("Cannot average empty vector"))?;
        let mut average_length = first_feat.length_distribution_dirty;
        let mut average_mat = first_feat.transition_matrix_dirty;
        let direction = first_feat.transition.reverse;
        for feat in iter {
            average_mat = average_mat + feat.transition_matrix_dirty;
            average_length = average_length + feat.length_distribution_dirty;
            len += 1;
        }

        // the error rate correction can make some value of the transition matrix negative
        // (shouldn't happen in theory, but that's life)
        // we fix those to 1e-4 (not 0, so that they are not blocked)
        // normalisation should take care of the rest.
        let sum = average_mat.clone().sum();
        average_mat.mapv_inplace(|a| if a < 0.0 { 1e-4 * sum } else { a });

        let transition = Arc::new(DNAMarkovChain::new(
            &(average_mat / (f64::from(len))),
            direction,
        )?);

        let average_feat = InsertionFeature::new(&(average_length / (f64::from(len))), transition)?;
        Ok(average_feat)
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug, PartialEq)]
pub struct InfEvent {
    pub v_index: usize,
    pub v_start_gene: usize, // start of the sequence in the V gene
    pub j_index: usize,
    pub j_start_seq: i64, // start of the palindromic J gene (with all dels) in the sequence
    pub d_index: usize,
    // position of the v,d,j genes in the sequence
    pub end_v: i64,
    pub start_d: i64,
    pub end_d: i64,
    pub start_j: i64,
    pub pos_d: i64,

    // sequences (only added after the inference is over)
    // need DnaLike container because : they can be aa sequence
    // & DnaLike is not a type that can work with pypi
    pub ins_vd: Option<DnaLike>,
    pub ins_dj: Option<DnaLike>,
    pub d_segment: Option<DnaLike>,
    pub sequence: Option<DnaLike>,
    pub junction: Option<DnaLike>,
    pub full_sequence: Option<Dna>,
    pub reconstructed_sequence: Option<Dna>,
    // likelihood (pgen + perror)
    pub likelihood: f64,
}

impl InfEvent {
    pub fn get_reconstructed_cdr3(self, model: &ModelVDJ) -> Result<Dna> {
        let seq = self.reconstructed_sequence.unwrap();
        let jgene = model.seg_js[self.j_index].clone();
        Ok(seq.extract_subsequence(
            model.seg_vs[self.v_index].cdr3_pos.unwrap(),
            seq.len() - jgene.seq.len() + jgene.cdr3_pos.unwrap() + 3,
        ))
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
#[derive(Clone, Debug)]
pub struct ResultInference {
    pub likelihood: f64,
    pub pgen: f64,
    pub best_event: Option<InfEvent>,
    // best_likelihood is more an useful tool during inference
    // than an actual likelihood of the best event
    // (the definition of "event" vary between model
    // best_event.likelihood is the way to go
    pub best_likelihood: f64,
    pub features: Option<Features>,
    pub human_readable: Option<ResultHuman>,
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
    pub fn get_likelihood_best_event(&self) -> Option<f64> {
        match self.get_best_event() {
            Some(x) => Some(x.likelihood),
            None => None,
        }
    }
}

/// A Result class that's easily readable
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all))]
pub struct ResultHuman {
    pub n_junction: String,
    pub aa_junction: String,
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
    pub d_name: String,
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl ResultInference {
    fn __repr__(&self) -> String {
        self.display().unwrap()
    }
    #[getter]
    pub fn get_best_v_gene(&self) -> String {
        self.human_readable.clone().unwrap().v_name
    }
    #[getter]
    pub fn get_best_j_gene(&self) -> String {
        self.human_readable.clone().unwrap().j_name
    }
    #[getter]
    pub fn get_best_d_gene(&self) -> String {
        self.human_readable.clone().unwrap().d_name
    }
    #[getter]
    pub fn get_reconstructed_sequence(&self) -> String {
        self.human_readable.clone().unwrap().reconstructed_seq
    }
}

impl ResultInference {
    pub fn display(&self) -> Result<String> {
        let rh = self.human_readable.clone();
        if self.best_event.is_none() || rh.is_none() {
            return Ok(format!(
                "Result:\n\
		 - Likelihood: {}\n\
		 - Pgen: {}\n",
                self.likelihood, self.pgen
            ));
        }
        let rh = rh.unwrap();
        Ok(format!(
            "Result:\n\
	     \tLikelihood: {:.2e}, pgen: {:.2e}\n\
	     \tMost likely event:\n\
	     \t- Junction (nucleotides): {} \n\
	     \t- Junction (amino acids): {} \n\
	     \t- V name: {} \n\
	     \t- J name: {} \n\
	     \t- likelihood ratio: {} \n ",
            self.likelihood,
            self.pgen,
            rh.n_junction,
            rh.aa_junction,
            rh.v_name,
            rh.j_name,
            rh.likelihood_ratio_best
        ))
    }

    /// Translate the result to an easier to read/print version
    pub fn load_human(&mut self, model: &vdj::Model) -> Result<()> {
        let best_event = self.get_best_event().ok_or(anyhow!("No event"))?;

        let translated_cdr3 = if best_event.junction.clone().unwrap().len() % 3 == 0 {
            best_event
                .junction
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

        self.human_readable = Some(ResultHuman {
            n_junction: best_event.junction.clone().unwrap().get_string(),
            aa_junction: translated_cdr3,
            likelihood: self.likelihood,
            pgen: self.pgen,
            likelihood_ratio_best: best_event.likelihood / self.likelihood,
            seq: best_event.sequence.clone().unwrap().get_string(),
            full_seq: best_event.full_sequence.clone().unwrap().get_string(),
            reconstructed_seq,
            aligned_v,
            aligned_j,
            v_name: model.get_v_gene(&best_event),
            d_name: model.get_d_gene(&best_event),
            j_name: model.get_j_gene(&best_event),
        });
        Ok(())
    }

    pub fn impossible() -> ResultInference {
        ResultInference {
            likelihood: 0.,
            pgen: 0.,
            best_event: None,
            best_likelihood: 0.,
            features: None,
            human_readable: None,
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
            let end_cdr3 = event.j_start_seq + cdr3_pos_j as i64 - model.range_del_j.0 + 3;

            event.junction = Some(
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

            let gene_d = model.seg_ds[event.d_index]
                .clone()
                .seq_with_pal
                .ok_or(anyhow!("Model not loaded correctly"))?;

            let gene_v_cut = DnaLike::from_dna(gene_v.extract_subsequence(0, event.v_start_gene));

            let mut full_seq = gene_v_cut.extended(&sequence.sequence);

            full_seq = full_seq.extended_with_dna(&gene_j.extract_subsequence(
                (sequence.sequence.len() as i64 - event.j_start_seq) as usize,
                gene_j.len(),
            ));

            event.full_sequence = Some(full_seq.to_dna());

            let mut reconstructed_seq =
                gene_v.extract_subsequence(0, (event.end_v + event.v_start_gene as i64) as usize);
            reconstructed_seq.extend(&event.ins_vd.clone().unwrap().to_dna());

            reconstructed_seq.extend(&gene_d.extract_subsequence(
                (-event.pos_d + event.start_d) as usize,
                (-event.pos_d + event.end_d) as usize,
            ));
            //            reconstructed_seq.extend(&event.d_segment.clone().unwrap());
            reconstructed_seq.extend(&event.ins_dj.clone().unwrap().to_dna());
            reconstructed_seq.extend(&gene_j.extract_padded_subsequence(
                event.start_j - event.j_start_seq,
                gene_j.len() as i64,
            ));
            event.reconstructed_sequence = Some(reconstructed_seq);
            self.best_event = Some(event);
            self.load_human(model)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
/// Generic "features" object that contains all the features of
/// the model
pub enum Features {
    VDJ(vdj::Features),
    VxDJ(v_dj::Features),
}

impl Features {
    pub fn infer(
        &mut self,
        sequence: &vdj::Sequence,
        ip: &InferenceParameters,
    ) -> Result<ResultInference> {
        match self {
            Features::VDJ(x) => x.infer(sequence, ip),
            Features::VxDJ(x) => x.infer(sequence, ip),
        }
    }

    pub fn normalize(&mut self) -> Result<()> {
        match self {
            Features::VDJ(x) => x.normalize(),
            Features::VxDJ(x) => x.normalize(),
        }
    }

    pub fn error(&self) -> &FeatureError {
        match self {
            Features::VDJ(x) => &x.error,
            Features::VxDJ(x) => &x.error,
        }
    }
    pub fn error_mut(&mut self) -> &mut FeatureError {
        match self {
            Features::VDJ(x) => &mut x.error,
            Features::VxDJ(x) => &mut x.error,
        }
    }

    pub fn update(
        features: Vec<Features>,
        model: &mut ModelVDJ,
        ip: &InferenceParameters,
    ) -> Result<(Vec<Features>, f64)> {
        Ok(match model.model_type {
            ModelStructure::VDJ => {
                let feats = vdj::Features::update(
                    features
                        .into_iter()
                        .filter_map(|x| {
                            if let Features::VDJ(f) = x {
                                Some(f)
                            } else {
                                None
                            }
                        })
                        .collect(),
                    model,
                    ip,
                )?;
                (feats.0.into_iter().map(Features::VDJ).collect(), feats.1)
            }
            ModelStructure::VxDJ => {
                let feats = v_dj::Features::update(
                    features
                        .into_iter()
                        .filter_map(|x| {
                            if let Features::VxDJ(f) = x {
                                Some(f)
                            } else {
                                None
                            }
                        })
                        .collect(),
                    model,
                    ip,
                )?;
                (feats.0.into_iter().map(Features::VxDJ).collect(), feats.1)
            }
        })
    }
}
