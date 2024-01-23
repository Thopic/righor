use crate::sequence::utils::difference_as_i64;
use crate::sequence::VJAlignment;
use crate::shared::feature::*;
use crate::shared::utils::insert_in_order;
use crate::shared::InferenceParameters;
use crate::vj::{Event, Model, Sequence, StaticEvent};
use anyhow::Result;
use itertools::iproduct;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all))]
pub struct Features {
    pub v: CategoricalFeature1,
    pub delv: CategoricalFeature1g1,
    pub j: CategoricalFeature1g1,
    pub delj: CategoricalFeature1g1,
    pub nb_insvj: CategoricalFeature1,
    pub insvj: InsertionFeature,
    pub error: ErrorSingleNucleotide,
}

impl Features {
    pub fn new(model: &Model, inference_params: &InferenceParameters) -> Result<Features> {
        Ok(Features {
            v: CategoricalFeature1::new(&model.p_v)?,
            delv: CategoricalFeature1g1::new(&model.p_del_v_given_v)?,
            j: CategoricalFeature1g1::new(&model.p_j_given_v)?,
            delj: CategoricalFeature1g1::new(&model.p_del_j_given_j)?,
            nb_insvj: CategoricalFeature1::new(&model.p_ins_vj)?,
            insvj: InsertionFeature::new(
                &model.p_ins_vj,
                &model.first_nt_bias_ins_vj,
                &model.markov_coefficients_vj,
            )?,
            error: ErrorSingleNucleotide::new(model.error_rate)?,
        })
    }

    /// Return an iterator over the Events
    fn range_event<'a>(
        &self,
        sequence: &'a Sequence,
    ) -> impl Iterator<Item = (&'a VJAlignment, &'a VJAlignment, usize, usize)> {
        iproduct!(
            sequence.v_genes.iter(),
            sequence.j_genes.iter(),
            0..self.delv.dim().0,
            0..self.delj.dim().0
        )
    }

    /// Estimate the likelihood of the event (without considering
    /// the insertions Markov model).
    fn likelihood_event(&self, e: &Event) -> f64 {
        // First check that nothing overlaps
        // v_end: position of the last element of the V gene + 1
        let v_end = difference_as_i64(e.v.end_seq, e.delv);
        let j_start = e.j.start_seq + e.delj;

        if v_end > (j_start as i64) {
            return 0.;
        }

        // Then compute the likelihood of each part (including insertion length)
        // We ignore the V/delV part (already computed)
        self.v.log_likelihood(e.v.index)
            + self.delv.log_likelihood((e.delv, e.v.index))
            + self.j.log_likelihood((e.j.index, e.v.index))
            + self.delj.log_likelihood((e.delj, e.j.index))
            + self
                .nb_insvj
                .log_likelihood((j_start as i64 - v_end) as usize)
            + self
                .error
                .log_likelihood((e.v.nb_errors(e.delv), e.v.length_with_deletion(e.delv)))
            + self
                .error
                .log_likelihood((e.j.nb_errors(e.delj), e.j.length_with_deletion(e.delj)))
    }

    /// Infer the likelihood of a specific events by looping over
    /// all possible recombinaison events.
    /// # Arguments:
    /// * `sequence`: Aligned DNA sequence
    /// * `inference_params`: Parameters used during the inference
    /// * `nb_best_events`: Number of most likely event to store
    /// # Returns:
    /// `(pgen, best_events)` where pgen is the likelihood of the sequence
    /// and best_events is a vector of tuples containing the most likely events and
    /// their probability.
    pub fn infer(
        &mut self,
        sequence: &Sequence,
        inference_params: &InferenceParameters,
        nb_best_events: usize,
    ) -> (f64, Vec<(f64, StaticEvent)>) {
        let mut probability_generation: f64 = 0.;
        let mut best_events = Vec::<(f64, StaticEvent)>::new();

        // Update all the marginals
        for (v, j, delv, delj) in self.range_event(sequence) {
            let e = Event { v, j, delv, delj };

            let lhood_vj = self.likelihood_event(&e);
            // drop that specific recombination event if the likelihood is too low
            if lhood_vj < inference_params.min_likelihood {
                continue;
            }

            // extract both inserted sequences
            let insvj = sequence.get_insertions_vj(&e);

            let l_total = lhood_vj + self.insvj.log_likelihood(&insvj);

            // drop that specific recombination event if the likelihood is too low
            if l_total < inference_params.min_likelihood {
                continue;
            }

            if nb_best_events > 0
                && ((best_events.len() < nb_best_events)
                    || (best_events.last().unwrap().0 < l_total))
            {
                best_events = insert_in_order(best_events, (l_total, e.to_static(insvj.clone())));
                best_events.truncate(nb_best_events);
            }

            probability_generation += l_total;

            // Update everything with the new likelihood
            self.v.dirty_update(v.index, l_total);
            self.j.dirty_update((j.index, v.index), l_total);
            self.delv.dirty_update((delv, v.index), l_total);
            self.delj.dirty_update((delj, j.index), l_total);
            self.nb_insvj.dirty_update(insvj.len(), l_total);
            self.insvj.dirty_update(&insvj, l_total);
            self.error.dirty_update(
                (
                    j.nb_errors(delj) + v.nb_errors(delv),
                    j.length_with_deletion(delj) + v.length_with_deletion(delv),
                ),
                l_total,
            );
        }
        return (probability_generation, best_events);
    }

    pub fn cleanup(&self) -> Result<Features> {
        // Compute the new marginals for the next round
        Ok(Features {
            v: self.v.cleanup()?,
            j: self.j.cleanup()?,
            delv: self.delv.cleanup()?,
            delj: self.delj.cleanup()?,
            nb_insvj: self.nb_insvj.cleanup()?,
            insvj: self.insvj.cleanup()?,
            error: self.error.cleanup()?,
        })
    }
}

#[cfg(not(feature = "py_binds"))]
impl Features {
    pub fn average(features: Vec<Features>) -> Result<Features> {
        Ok(Features {
            v: CategoricalFeature1::average(features.iter().map(|a| a.v.clone()))?,
            delv: CategoricalFeature1g1::average(features.iter().map(|a| a.delv.clone()))?,
            j: CategoricalFeature1g1::average(features.iter().map(|a| a.j.clone()))?,
            delj: CategoricalFeature1g1::average(features.iter().map(|a| a.delj.clone()))?,
            nb_insvj: CategoricalFeature1::average(features.iter().map(|a| a.nb_insvj.clone()))?,
            insvj: InsertionFeature::average(features.iter().map(|a| a.insvj.clone()))?,
            error: ErrorSingleNucleotide::average(features.iter().map(|a| a.error.clone()))?,
        })
    }
}

#[cfg(feature = "py_binds")]
#[pymethods]
impl Features {
    #[staticmethod]
    pub fn average(features: Vec<Features>) -> Result<Features> {
        Ok(Features {
            v: CategoricalFeature1::average(features.iter().map(|a| a.v.clone()))?,
            delv: CategoricalFeature1g1::average(features.iter().map(|a| a.delv.clone()))?,
            j: CategoricalFeature1g1::average(features.iter().map(|a| a.j.clone()))?,
            delj: CategoricalFeature1g1::average(features.iter().map(|a| a.delj.clone()))?,
            nb_insvj: CategoricalFeature1::average(features.iter().map(|a| a.nb_insvj.clone()))?,
            insvj: MarkovFeature::average(features.iter().map(|a| a.insvj.clone()))?,
            error: ErrorSingleNucleotide::average(features.iter().map(|a| a.error.clone()))?,
        })
    }
}
