/// Contains all the error models defined and their features (for inference)
use crate::shared::distributions::{HistogramDistribution, UniformError};
use crate::shared::feature::Feature;

use crate::shared::Dna;
use crate::shared::ErrorAlignment;
use crate::shared::StaticEvent;
use anyhow::{anyhow, Result};

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

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

    /// return a new feature
    pub fn get_feature(&self) -> Result<FeatureError> {
        match self {
            ErrorParameters::ConstantRate(err) => {
                Ok(FeatureError::ConstantRate(err.get_feature()?))
            }
            ErrorParameters::UniformRate(err) => Ok(FeatureError::UniformRate(err.get_feature()?)),
        }
    }

    /// Update the model from a list of errors
    pub fn update_error(
        features: Vec<FeatureError>,
        model: &mut ErrorParameters,
    ) -> Result<Vec<FeatureError>> {
        Ok(match model {
            ErrorParameters::ConstantRate(m) => ErrorConstantRate::update_error(
                features
                    .into_iter()
                    .filter_map(|el| el.try_into().ok())
                    .collect(),
                m,
            )?
            .into_iter()
            .map(FeatureError::ConstantRate)
            .collect(),
            ErrorParameters::UniformRate(m) => ErrorUniformRate::update_error(
                features
                    .into_iter()
                    .filter_map(|el| el.try_into().ok())
                    .collect(),
                m,
            )?
            .into_iter()
            .map(FeatureError::UniformRate)
            .collect(),
        })
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

    pub fn update_error(
        features: Vec<FeatureErrorConstant>,
        error: &mut ErrorConstantRate,
    ) -> Result<Vec<FeatureErrorConstant>> {
        let mut len = 1;
        let mut iter = features.iter().clone();
        let first_feat = iter.next().ok_or(anyhow!("Cannot average empty vector"))?;
        let mut sum_err = first_feat.total_errors_dirty;
        let mut sum_length = first_feat.total_lengths_dirty;
        for feat in iter {
            sum_err += feat.total_errors_dirty;
            sum_length += feat.total_lengths_dirty;
            len += 1;
        }
        let error_rate = if sum_length == 0. {
            sum_err / sum_length
        } else {
            0.
        };
        // update the error model
        *error = ErrorConstantRate::new(error_rate);
        // return the features
        let constant_feat = FeatureErrorConstant::new(error_rate)?;
        Ok(vec![constant_feat; len])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Uniform mutation rate along the sequence, but
/// the error rate can differ from one sequence to the next
pub struct ErrorUniformRate {
    // The two vector here are not used to infer anything, but
    // they keep the histogram of values somewhere in memory
    /// Bins of the histogram of the error rate
    pub bins: Vec<f64>,
    /// Probabilities of the histogram of the error rate
    pub probas: Vec<f64>,

    #[serde(skip)]
    error_rate_gen: HistogramDistribution,
    #[serde(skip)]
    gen: UniformError,
}

impl Default for ErrorUniformRate {
    fn default() -> ErrorUniformRate {
        // default distribution (a bit ad-hoc but well)
        ErrorUniformRate::new(
            (0..1001).map(|x| (x as f64) / 1000.).collect(),
            vec![1. / 1000.; 1000],
        )
        .unwrap()
    }
}

impl ErrorUniformRate {
    pub fn new(bins: Vec<f64>, probas: Vec<f64>) -> Result<ErrorUniformRate> {
        let mut error = ErrorUniformRate {
            bins: bins,
            probas: probas,
            error_rate_gen: HistogramDistribution::default(),
            gen: UniformError::new(),
        };

        error.init_generation()?;
        Ok(error)
    }

    pub fn init_generation(&mut self) -> Result<()> {
        self.error_rate_gen = HistogramDistribution::new(self.bins.clone(), self.probas.clone())?;
        Ok(())
    }

    fn apply_to_sequence<R: Rng>(&self, full_seq: &Dna, event: &mut StaticEvent, rng: &mut R) {
        let effective_error_rate = self.error_rate_gen.generate(rng) * 4. / 3.;
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

        ErrorUniformRate::new(bins, probas)
    }

    fn uniform(&self) -> Result<ErrorUniformRate> {
        ErrorUniformRate::new(
            self.bins.clone(),
            vec![1. / (self.bins.len() as f64 - 1.); self.bins.len() - 1],
        )
    }

    fn similar(e1: Self, e2: Self) -> bool {
        if e1.bins.len() != e2.bins.len() || e1.probas.len() != e2.probas.len() {
            return false;
        }

        for i in 0..e1.probas.len() {
            if (e1.bins[i] - e2.bins[i]).abs() > 1e-4 {
                return false;
            }
            if (e1.bins[i + 1] - e2.bins[i + 1]).abs() > 1e-4 {
                return false;
            }
            if (e1.probas[i] - e2.probas[i]).abs() > 1e-4 {
                return false;
            }
        }
        true
    }
    pub fn get_feature(&self) -> Result<FeatureErrorUniform> {
        // by default return the average of the distribution
        FeatureErrorUniform::new(
            self.probas
                .iter()
                .enumerate()
                .map(|(i, pi)| pi * (self.bins[i] + self.bins[i + 1]) / 2.)
                .sum(),
        )
    }

    fn update_error(
        features: Vec<FeatureErrorUniform>,
        error: &mut ErrorUniformRate,
    ) -> Result<Vec<FeatureErrorUniform>> {
        let mut counts = vec![0usize; error.bins.len()];
        for feat in &features {
            let error_rate = if feat.total_lengths_dirty == 0. {
                0.
            } else {
                feat.total_errors_dirty / feat.total_lengths_dirty
            };
            let idx = error
                .bins
                .binary_search_by(|&bin| bin.partial_cmp(&error_rate).unwrap());
            match idx {
                Ok(i) => counts[i] += 1,
                Err(i) => counts[i - 1] += 1,
            }
        }

        // Convert counts to probabilities
        let total = features.len() as f64;
        let probas: Vec<f64> = counts.iter().map(|&count| count as f64 / total).collect();

        error.probas = probas;
        error.init_generation()?;

        // return the vector without modifying it
        Ok(features)
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

impl TryFrom<FeatureError> for FeatureErrorConstant {
    type Error = anyhow::Error;
    fn try_from(value: FeatureError) -> Result<Self> {
        if let FeatureError::ConstantRate(v) = value {
            Ok(v)
        } else {
            Err(anyhow!("Wrong feature type"))
        }
    }
}

impl TryFrom<FeatureError> for FeatureErrorUniform {
    type Error = anyhow::Error;
    fn try_from(value: FeatureError) -> Result<Self> {
        if let FeatureError::UniformRate(v) = value {
            Ok(v)
        } else {
            Err(anyhow!("Wrong feature type"))
        }
    }
}

impl FeatureError {
    // pub fn get_parameters(&self) -> Result<ErrorParameters> {
    //     Ok(match self {
    //         FeatureError::ConstantRate(x) => ErrorParameters::ConstantRate(x.get_parameters()?),
    //         FeatureError::UniformRate(x) => ErrorParameters::UniformRate(x.get_parameters()?),
    //     })
    // }

    // pub fn average(iter: impl Iterator<Item = FeatureError> + Clone) -> Result<Vec<FeatureError>> {
    //     let avg_constants = FeatureErrorConstant::average(iter.clone().filter_map(|e| {
    //         if let FeatureError::ConstantRate(fec) = e {
    //             Some(fec)
    //         } else {
    //             None
    //         }
    //     }))?;

    //     let avg_uniforms = FeatureErrorUniform::average(iter.filter_map(|e| {
    //         if let FeatureError::UniformRate(feu) = e {
    //             Some(feu)
    //         } else {
    //             None
    //         }
    //     }))?;

    //     if !avg_uniforms.is_empty() && !avg_constants.is_empty() {
    //         return Err(anyhow!(
    //             "Multiple error models in the same model, this shouldn't happen."
    //         ));
    //     }

    //     let result: Vec<FeatureError> = avg_constants
    //         .into_iter()
    //         .map(FeatureError::ConstantRate)
    //         .chain(avg_uniforms.into_iter().map(FeatureError::UniformRate))
    //         .collect();

    //     Ok(result)
    // }

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
        println!("{:?}", error_rate);
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

impl Feature<ErrorAlignment> for FeatureErrorUniform {
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

    fn scale_dirty(&mut self, _factor: f64) {}
}

impl FeatureErrorUniform {
    pub fn new(error_rate: f64) -> Result<FeatureErrorUniform> {
        if !(0. ..1.).contains(&error_rate) || (error_rate.is_nan()) || (error_rate.is_infinite()) {
            return Err(anyhow!(
                "Error in FeatureErrorConstant Feature creation. Negative/NaN/infinite error rate."
            ));
        }
        Ok(FeatureErrorUniform {
            error_rate,
            logrs3: (error_rate / 3.).log2(),
            log1mr: (1. - error_rate).log2(),
            total_lengths_dirty: 0.,
            total_errors_dirty: 0.,
            total_probas_dirty: 0.,
        })
    }

    pub fn get_error_rate(&self) -> f64 {
        self.error_rate
    }
}
