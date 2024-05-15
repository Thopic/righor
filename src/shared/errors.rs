/// Contains all the error models defined and their features (for inference)
use crate::shared::distributions::{DiscreteDistribution, UniformError};
use crate::shared::feature::Feature;
use crate::shared::utils::Normalize;
use crate::shared::Dna;
use crate::shared::ErrorAlignment;
use crate::shared::StaticEvent;
use anyhow::{anyhow, Result};
use memoize::memoize;
use ndarray::Array1;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::hash;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ErrorParameters {
    ConstantRate(ErrorConstantRate),
    UniformRate(ErrorUniformRate),
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pyclass(name = "ErrorParameters")]
#[derive(Clone, Debug, Default)]
pub struct PyErrorParameters {
    pub s: ErrorParameters,
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl PyErrorParameters {
    fn __repr__(&self) -> String {
        match &self.s {
            ErrorParameters::ConstantRate(x) => {
                format!(
                    "Constant Error model [same error rate for all sequences]. Error rate = {}",
                    x.error_rate
                )
            }
            ErrorParameters::UniformRate(x) => {
                format!(
                    "Uniform Error model [each sequence can have a ≠ number of mutations, \
                        distributed uniformly along the sequence].\
                        \nAverage error rate\
                        = {}",
                    x.probas
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| p * (x.bins[i] + x.bins[i + 1]) / 2.0
                            * (x.bins[i + 1] - x.bins[i]))
                        .sum::<f64>()
                )
            }
        }
    }

    #[staticmethod]
    fn uniform_error(probas: Vec<f64>, bins: Vec<f64>) -> PyResult<PyErrorParameters> {
        let error = ErrorUniformRate::new(probas, bins)?;
        Ok(PyErrorParameters {
            s: ErrorParameters::UniformRate(error),
        })
    }

    #[staticmethod]
    fn constant_error(error_rate: f64) -> PyResult<PyErrorParameters> {
        Ok(PyErrorParameters {
            s: ErrorParameters::ConstantRate(ErrorConstantRate::new(error_rate)),
        })
    }

    #[getter]
    fn get_error_rate(&self) -> PyResult<f64> {
        match &self.s {
            ErrorParameters::ConstantRate(x) => Ok(x.error_rate),
            ErrorParameters::UniformRate(_) => {
                Err(anyhow!("No generic error rate in an uniform Error model."))?
            }
        }
    }

    #[getter]
    fn get_probability_distribution(&self) -> PyResult<(Vec<f64>, Vec<f64>)> {
        match &self.s {
            ErrorParameters::ConstantRate(_) => Err(anyhow!(
                "No (stored) number error distribution in a constant error-rate Error model."
            ))?,
            ErrorParameters::UniformRate(x) => Ok((x.bins.clone(), x.probas.clone())),
        }
    }
}

impl Default for ErrorParameters {
    fn default() -> ErrorParameters {
        return ErrorParameters::ConstantRate(ErrorConstantRate::new(0.));
    }
}

impl ErrorParameters {
    /// Apply the error to the generated sequence
    pub fn apply_to_sequence<R: Rng>(
        &mut self,
        full_seq: &Dna,
        event: &mut StaticEvent,
        rng: &mut R,
    ) {
        match self {
            ErrorParameters::ConstantRate(err) => err.apply_to_sequence(full_seq, event, rng),
            ErrorParameters::UniformRate(err) => err.apply_to_sequence(full_seq, event, rng),
        };
    }

    /// write the error in igor format
    pub fn write(&self) -> String {
        match self {
            ErrorParameters::ConstantRate(err) => err.write(),
            ErrorParameters::UniformRate(err) => err.write(),
        }
    }

    pub fn no_error(&self) -> bool {
        match self {
            ErrorParameters::ConstantRate(err) => err.no_error(),
            ErrorParameters::UniformRate(_) => false,
        }
    }

    /// Set the error to 0
    pub fn remove_error(&mut self) {
        *self = ErrorParameters::ConstantRate(ErrorConstantRate::new(0.));
    }

    /// Uniform value (default value pre-inference)
    pub fn uniform(&self) -> Result<ErrorParameters> {
        Ok(match self {
            ErrorParameters::ConstantRate(x) => {
                ErrorParameters::ConstantRate(ErrorConstantRate::uniform(x)?)
            }
            ErrorParameters::UniformRate(x) => {
                ErrorParameters::UniformRate(ErrorUniformRate::uniform(x)?)
            }
        })
    }

    /// Check if two error parameters are close enough
    pub fn similar(e1: ErrorParameters, e2: ErrorParameters) -> bool {
        match e1 {
            ErrorParameters::ConstantRate(ee1) => match e2 {
                ErrorParameters::ConstantRate(ee2) => ErrorConstantRate::similar(ee1, ee2),
                ErrorParameters::UniformRate(_) => false,
            },
            ErrorParameters::UniformRate(ee1) => match e2 {
                ErrorParameters::ConstantRate(_) => false,
                ErrorParameters::UniformRate(ee2) => ErrorUniformRate::similar(ee1, ee2),
            },
        }
    }

    /// return the feature
    pub fn get_feature(&self) -> Result<FeatureError> {
        match self {
            ErrorParameters::ConstantRate(err) => {
                Ok(FeatureError::ConstantRate(err.get_feature()?))
            }
            ErrorParameters::UniformRate(err) => Ok(FeatureError::UniformRate(err.get_feature()?)),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Most basic error model
pub struct ErrorConstantRate {
    pub error_rate: f64,
    #[serde(skip)]
    gen: UniformError,
}

impl Default for ErrorConstantRate {
    fn default() -> ErrorConstantRate {
        ErrorConstantRate::new(0.)
    }
}

impl ErrorConstantRate {
    pub fn new(error_rate: f64) -> ErrorConstantRate {
        ErrorConstantRate {
            error_rate: error_rate,
            gen: UniformError::new(),
        }
    }

    fn apply_to_sequence<R: Rng>(&self, full_seq: &Dna, event: &mut StaticEvent, rng: &mut R) {
        let effective_error_rate = self.error_rate * 4. / 3.;
        let mut errors =
            Vec::with_capacity((effective_error_rate * full_seq.len() as f64).ceil() as usize);

        for (idx, nucleotide) in full_seq.seq.iter().enumerate() {
            if self.gen.is_error(effective_error_rate, rng) {
                let a = self.gen.random_nucleotide(rng);
                if a != *nucleotide {
                    errors.push((idx, a));
                }
            }
        }
        event.set_errors(errors);
    }

    fn write(&self) -> String {
        format!(
            "@ErrorRate\n\
	     #SingleErrorRate\n\
	     {}\n",
            self.error_rate
        )
    }

    fn no_error(&self) -> bool {
        self.error_rate == 0.
    }

    fn uniform(&self) -> Result<ErrorConstantRate> {
        Ok(ErrorConstantRate::new(0.1))
    }

    fn similar(e1: Self, e2: Self) -> bool {
        (e1.error_rate - e2.error_rate).abs() < 1e-4
    }
    pub fn get_feature(&self) -> Result<FeatureErrorConstant> {
        FeatureErrorConstant::new(self.error_rate)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Uniform mutation rate along the sequence, but
/// the average number of mutations of each sequence can be different
pub struct ErrorUniformRate {
    /// Bins of the histogram of the error rate
    pub bins: Vec<f64>,
    /// Probabilities of the histogram of the error rate
    pub probas: Vec<f64>,
    #[serde(skip)]
    distribution: DiscreteDistribution,
    #[serde(skip)]
    gen: UniformError,
}

impl Default for ErrorUniformRate {
    fn default() -> ErrorUniformRate {
        // default distribution (a bit ad-hoc but well)
        ErrorUniformRate::new(
            vec![1. / 100., 100.],
            (0..101).map(|x| (x as f64) / 100. * 0.6).collect(),
        )
        .unwrap()
    }
}

impl ErrorUniformRate {
    pub fn new(probas: Vec<f64>, bins: Vec<f64>) -> Result<ErrorUniformRate> {
        Ok(ErrorUniformRate {
            probas: probas.clone(),
            bins,
            distribution: DiscreteDistribution::new(probas)?,
            gen: UniformError::new(),
        })
    }

    fn apply_to_sequence<R: Rng>(&mut self, full_seq: &Dna, event: &mut StaticEvent, rng: &mut R) {
        let error_bin = self.distribution.generate(rng);
        let effective_error_rate = (self.bins[error_bin] + self.bins[error_bin + 1]) / 2. * 4. / 3.;
        let mut errors =
            Vec::with_capacity((effective_error_rate * full_seq.len() as f64).ceil() as usize);

        for (idx, nucleotide) in full_seq.seq.iter().enumerate() {
            if self.gen.is_error(effective_error_rate, rng) {
                let a = self.gen.random_nucleotide(rng);
                if a != *nucleotide {
                    errors.push((idx, a));
                }
            }
        }
        event.set_errors(errors);
    }

    fn write(&self) -> String {
        format!(
            "@ErrorRate\n\
         #IndividualErrorRate\n\
         {}",
            self.probas
                .iter()
                .enumerate()
                .map(|(i, &p)| format!("%{};{};{}", self.bins[i], self.bins[i + 1], p))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    pub fn load(str_vec: &Vec<String>) -> Result<ErrorUniformRate> {
        if !str_vec[1].starts_with("#IndividualErrorRate") {
            return Err(anyhow!("Wrong error type"));
        }
        let mut bins = Vec::new();
        let mut probas = Vec::new();

        for s in str_vec.iter().skip(2) {
            let parts: Vec<&str> = s[1..].split(';').collect();
            let bin_start: f64 = parts[0].parse().unwrap();
            let bin_end: f64 = parts[1].parse().unwrap();
            let proba: f64 = parts[2].parse().unwrap();
            bins.push(bin_start);
            bins.push(bin_end);
            probas.push(proba);
        }
        bins.dedup();
        ErrorUniformRate::new(probas, bins)
    }

    fn uniform(&self) -> Result<ErrorUniformRate> {
        ErrorUniformRate::new(
            vec![1. / ((self.bins.len() - 1) as f64); self.bins.len() - 1],
            self.bins.clone(),
        )
    }

    fn similar(e1: Self, e2: Self) -> bool {
        if e1.bins.len() != e2.bins.len() {
            return false;
        }
        for ii in 0..e1.probas.len() {
            if (e1.bins[ii] - e2.bins[ii]).abs() > 1e-4 {
                return false;
            }
            if (e1.bins[ii + 1] - e2.bins[ii + 1]).abs() > 1e-4 {
                return false;
            }
            if ((e1.probas[ii] - e2.probas[ii]) / (e1.probas[ii] + e2.probas[ii])).abs() > 1e-4 {
                return false;
            }
        }
        true
    }

    pub fn get_feature(&self) -> Result<FeatureErrorUniform> {
        FeatureErrorUniform::new(&Array1::from(self.probas.clone()), self.bins.clone())
    }
}

#[derive(Clone, Debug)]
pub enum FeatureError {
    ConstantRate(FeatureErrorConstant),
    UniformRate(FeatureErrorUniform),
}

impl Default for FeatureError {
    fn default() -> FeatureError {
        FeatureError::ConstantRate(FeatureErrorConstant::default())
    }
}

impl FeatureError {
    pub fn get_parameters(&self) -> Result<ErrorParameters> {
        Ok(match self {
            FeatureError::ConstantRate(x) => ErrorParameters::ConstantRate(x.get_parameters()?),
            FeatureError::UniformRate(x) => ErrorParameters::UniformRate(x.get_parameters()?),
        })
    }

    pub fn average(iter: impl Iterator<Item = FeatureError> + Clone) -> Result<FeatureError> {
        let mut cloned_iter = iter.clone();
        match cloned_iter.next() {
            Some(first_item) => Ok(match first_item {
                FeatureError::ConstantRate(_) => FeatureError::ConstantRate(
                    FeatureErrorConstant::average(iter.filter_map(|item| {
                        if let FeatureError::ConstantRate(a) = item {
                            Some(a)
                        } else {
                            None
                        }
                    }))?,
                ),
                FeatureError::UniformRate(_) => FeatureError::UniformRate(
                    FeatureErrorUniform::average(iter.filter_map(|item| {
                        if let FeatureError::UniformRate(a) = item {
                            Some(a)
                        } else {
                            None
                        }
                    }))?,
                ),
            }),
            None => {
                // Handle empty iterator case
                Err(anyhow!("Cannot average empty vector"))
            }
        }
    }

    pub fn scale_dirty(&mut self, factor: f64) {
        match self {
            FeatureError::ConstantRate(f) => f.scale_dirty(factor),
            FeatureError::UniformRate(f) => f.scale_dirty(factor),
        }
    }

    pub fn remove_error(&mut self) -> Result<()> {
        *self = FeatureError::ConstantRate(FeatureErrorConstant::new(0.)?);
        Ok(())
    }

    pub fn likelihood(&self, observation: ErrorAlignment) -> f64 {
        match self {
            FeatureError::ConstantRate(f) => f.likelihood(observation),
            FeatureError::UniformRate(f) => f.likelihood(observation),
        }
    }

    pub fn dirty_update(&mut self, observation: ErrorAlignment, likelihood: f64) {
        match self {
            FeatureError::ConstantRate(f) => f.dirty_update(observation, likelihood),
            FeatureError::UniformRate(f) => f.dirty_update(observation, likelihood),
        }
    }
}

// Most basic error feature
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
pub struct FeatureErrorConstant {
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

impl FeatureErrorConstant {
    pub fn new(error_rate: f64) -> Result<FeatureErrorConstant> {
        if !(0. ..1.).contains(&error_rate) || (error_rate.is_nan()) || (error_rate.is_infinite()) {
            return Err(anyhow!(
                "Error in FeatureErrorConstant Feature creation. Negative/NaN/infinite error rate."
            ));
        }
        Ok(FeatureErrorConstant {
            error_rate,
            logrs3: (error_rate / 3.).log2(),
            log1mr: (1. - error_rate).log2(),
            total_lengths_dirty: 0.,
            total_errors_dirty: 0.,
            total_probas_dirty: 0.,
        })
    }
}

impl Feature<ErrorAlignment> for FeatureErrorConstant {
    /// Arguments
    /// - observation: "(nb of error, length of the sequence without insertion)"
    /// - likelihood: measured likelihood of the event
    fn dirty_update(&mut self, observation: ErrorAlignment, likelihood: f64) {
        self.total_lengths_dirty += likelihood * (observation.sequence_length as f64);
        self.total_errors_dirty += likelihood * (observation.nb_errors as f64);
        self.total_probas_dirty += likelihood;
    }

    /// Arguments
    /// - observation: "(nb of error, length of the sequence without insertion)"
    /// The complete formula is likelihood = (r/3)^(nb error) * (1-r)^(length - nb error)
    fn likelihood(&self, observation: ErrorAlignment) -> f64 {
        if observation.nb_errors == 0 {
            return (observation.sequence_length as f64 * self.log1mr).exp2();
        }
        ((observation.nb_errors as f64) * self.logrs3
            + ((observation.sequence_length - observation.nb_errors) as f64) * self.log1mr)
            .exp2()
    }

    fn scale_dirty(&mut self, factor: f64) {
        self.total_errors_dirty *= factor;
        self.total_lengths_dirty *= factor;
    }

    fn average(
        mut iter: impl Iterator<Item = FeatureErrorConstant> + Clone,
    ) -> Result<FeatureErrorConstant> {
        let first_feat = iter.next().ok_or(anyhow!("Cannot average empty vector"))?;
        let mut sum_err = first_feat.total_errors_dirty;
        let mut sum_length = first_feat.total_lengths_dirty;
        for feat in iter {
            sum_err += feat.total_errors_dirty;
            sum_length += feat.total_lengths_dirty;
        }
        if sum_length == 0. {
            return FeatureErrorConstant::new(0.);
        }
        FeatureErrorConstant::new(sum_err / sum_length)
    }
}

impl FeatureErrorConstant {
    pub fn remove_error(&mut self) -> Result<()> {
        *self = FeatureErrorConstant::new(0.)?;
        Ok(())
    }

    pub fn get_parameters(&self) -> Result<ErrorConstantRate> {
        Ok(ErrorConstantRate::new(self.error_rate))
    }
}

#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
/// Uniform error rate on the sequence, but the error rate depends on the sequence
pub struct FeatureErrorUniform {
    /// histogram bins to infer the continuous distribution
    pub bins: Vec<f64>,
    /// The probability for the error rate to be in one of the bin
    pub probas: Array1<f64>,

    probas_dirty: Array1<f64>,

    /// used to scale the nucleotide transition matrix for the insertions
    total_lengths_dirty: f64,
    total_errors_dirty: f64,
}

impl Feature<ErrorAlignment> for FeatureErrorUniform {
    fn dirty_update(&mut self, observation: ErrorAlignment, likelihood: f64) {
        self.total_lengths_dirty += likelihood * (observation.sequence_length as f64);
        self.total_errors_dirty += likelihood * (observation.nb_errors as f64);

        let mut p = 0.;
        for (i, pi) in self.probas.iter().enumerate() {
            let r = (self.bins[i + 1] + self.bins[i]) / 2.;
            p += pi
                * (r / 3.).powi(observation.nb_errors as i32)
                * (1. - r).powi((observation.sequence_length - observation.nb_errors) as i32);
        }

        for (i, pi) in self.probas.iter().enumerate() {
            let r = (self.bins[i + 1] + self.bins[i]) / 2.;
            self.probas_dirty[i] += likelihood
                * pi
                * (r / 3.).powi(observation.nb_errors as i32)
                * (1. - r).powi((observation.sequence_length - observation.nb_errors) as i32)
                / p;
        }
    }

    fn likelihood(&self, observation: ErrorAlignment) -> f64 {
        let mut p = 0.;
        for (i, pi) in self.probas.iter().enumerate() {
            let ri = (self.bins[i + 1] + self.bins[i]) / 2.;
            p += pi
                * poisson_proba(
                    Hashablef64(ri),
                    observation.nb_errors as i32,
                    observation.sequence_length as i32,
                )
        }
        p
    }

    fn scale_dirty(&mut self, factor: f64) {
        self.total_errors_dirty *= factor;
        self.total_lengths_dirty *= factor;
        self.probas *= factor
    }

    fn average(
        mut iter: impl Iterator<Item = FeatureErrorUniform> + Clone,
    ) -> Result<FeatureErrorUniform> {
        let first = iter
            .clone()
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?;
        let mut average_probas = iter
            .next()
            .ok_or(anyhow!("Cannot average empty vector"))?
            .probas_dirty;
        let mut len = 0;
        for feat in iter {
            average_probas = average_probas + feat.probas_dirty;
            len += 1;
        }

        FeatureErrorUniform::new(&(average_probas / (len as f64)), first.bins)
    }
}

#[derive(Debug, Copy, Clone)]
struct Hashablef64(f64);

impl PartialEq for Hashablef64 {
    fn eq(&self, other: &Hashablef64) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl hash::Hash for Hashablef64 {
    fn hash<H>(&self, state: &mut H)
    where
        H: hash::Hasher,
    {
        self.0.to_bits().hash(state)
    }
}

impl Eq for Hashablef64 {}

#[memoize]
fn poisson_proba(r: Hashablef64, n: i32, l: i32) -> f64 {
    (r.0 / 3.).powi(n) * (1. - r.0).powi(l - n)
}

impl FeatureErrorUniform {
    pub fn new(probabilities: &Array1<f64>, bins: Vec<f64>) -> Result<FeatureErrorUniform> {
        Ok(FeatureErrorUniform {
            probas_dirty: Array1::<f64>::zeros(probabilities.dim()),
            bins,
            probas: probabilities.normalize_distribution()?,
            total_errors_dirty: 0.,
            total_lengths_dirty: 0.,
        })
    }

    pub fn get_parameters(&self) -> Result<ErrorUniformRate> {
        ErrorUniformRate::new(self.probas.to_vec(), self.bins.clone())
    }

    pub fn get_error_rate(&self) -> f64 {
        if self.total_lengths_dirty == 0. {
            return 0.;
        }
        self.total_errors_dirty / self.total_lengths_dirty
    }
}
