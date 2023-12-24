use crate::feature::*;
use crate::model::ModelVDJ;
use crate::sequence::{DAlignment, EventVDJ, SequenceVDJ, VJAlignment};
use crate::utils_sequences::difference_as_i64;
use anyhow::Result;
use itertools::iproduct;
use pyo3::{pyclass, pyfunction, pymethods};

#[derive(Default, Clone, Debug)]
#[pyclass(get_all, set_all)]
pub struct InferenceParameters {
    pub min_likelihood_error: f64,
    pub min_likelihood: f64,
    pub nb_rounds_em: usize,
}

#[pymethods]
impl InferenceParameters {
    #[new]
    pub fn new(min_likelihood_error: f64, min_likelihood: f64, nb_rounds_em: usize) -> Self {
        Self {
            min_likelihood_error,
            min_likelihood,
            nb_rounds_em,
        }
    }
}

#[derive(Default, Clone, Debug)]
#[pyclass(get_all)]
pub struct FeaturesVDJ {
    v: CategoricalFeature1,
    delv: CategoricalFeature1g1,
    dj: CategoricalFeature2,
    delj: CategoricalFeature1g1,
    deld: CategoricalFeature2g1,
    nb_insvd: CategoricalFeature1,
    nb_insdj: CategoricalFeature1,
    insvd: MarkovFeature,
    insdj: MarkovFeature,
    error: ErrorPoisson,
}

impl FeaturesVDJ {
    pub fn new(model: &ModelVDJ, inference_params: &InferenceParameters) -> Result<FeaturesVDJ> {
        Ok(FeaturesVDJ {
            v: CategoricalFeature1::new(&model.p_v)?,
            delv: CategoricalFeature1g1::new(&model.p_del_v_given_v)?,
            dj: CategoricalFeature2::new(&model.p_dj)?,
            delj: CategoricalFeature1g1::new(&model.p_del_j_given_j)?,
            deld: CategoricalFeature2g1::new(&model.p_del_d3_del_d5)?, // dim: (d3, d5, d)
            nb_insvd: CategoricalFeature1::new(&model.p_ins_vd)?,
            nb_insdj: CategoricalFeature1::new(&model.p_ins_dj)?,
            insvd: MarkovFeature::new(&model.first_nt_bias_ins_vd, &model.markov_coefficients_vd)?,
            insdj: MarkovFeature::new(&model.first_nt_bias_ins_dj, &model.markov_coefficients_dj)?,
            error: ErrorPoisson::new(model.error_rate, inference_params.min_likelihood_error)?,
        })
    }

    pub fn update_model(&self, model: &mut ModelVDJ) {
        model.p_v = self.v.probas.clone();
        model.p_del_v_given_v = self.delv.probas.clone();
        model.p_dj = self.dj.probas.clone();
        model.p_del_j_given_j = self.delj.probas.clone();
        model.p_del_d3_del_d5 = self.deld.probas.clone();
        model.p_ins_vd = self.nb_insvd.probas.clone();
        model.p_ins_dj = self.nb_insdj.probas.clone();
        (model.first_nt_bias_ins_vd, model.markov_coefficients_vd) = self.insvd.get_parameters();
        (model.first_nt_bias_ins_dj, model.markov_coefficients_dj) = self.insvd.get_parameters();
        model.error_rate = self.error.error_rate;
    }

    // Return an iterator over V & delV
    fn range_v<'a>(
        &self,
        sequence: &'a SequenceVDJ,
    ) -> impl Iterator<Item = (&'a VJAlignment, usize)> {
        iproduct!(sequence.v_genes.iter(), 0..self.delv.dim().0)
    }

    // Return an iterator over Events
    fn range_dj<'a>(
        &self,
        sequence: &'a SequenceVDJ,
    ) -> impl Iterator<Item = (&'a VJAlignment, usize, &'a DAlignment, usize, usize)> {
        iproduct!(
            sequence.j_genes.iter(),
            0..self.delj.dim().0,
            sequence.d_genes.iter(),
            0..self.deld.dim().1,
            0..self.deld.dim().0
        )
    }

    fn likelihood_v(&self, v: &VJAlignment, delv: usize) -> f64 {
        self.v.likelihood(v.index)
            * self.delv.likelihood((delv, v.index))
            * self.error.likelihood(v.nb_errors(delv))
    }

    fn likelihood_dj(&self, e: &EventVDJ) -> f64 {
        // Estimate the likelihood of the d/j portion of the alignment
        // First check that nothing overlaps
        let v_end = difference_as_i64(e.v.end_seq, e.delv);
        let d_start = e.d.pos + e.deld5;
        let d_end = e.d.pos + e.d.len() - e.deld3;
        let j_start = e.j.start_seq + e.delj;

        if (v_end > (d_start as i64)) | (d_start > d_end) | (d_end > j_start) {
            return 0.;
        }

        // Then compute the likelihood of each part (including insertion length)
        // We ignore the V/delV part (already computed)
        self.dj.likelihood((e.d.index, e.j.index))
            * self.delj.likelihood((e.delj, e.j.index))
            * self.deld.likelihood((e.deld3, e.deld5, e.d.index))
            * self.nb_insvd.likelihood((d_start as i64 - v_end) as usize)
            * self.nb_insdj.likelihood(j_start - d_end)
            * self.error.likelihood(e.d.nb_errors(e.deld3, e.deld5))
            * self.error.likelihood(e.j.nb_errors(e.delj))
    }

    fn dirty_update(&mut self, sequence: &SequenceVDJ, inference_params: &InferenceParameters) {
        // Update all the marginals
        for (v, delv) in self.range_v(sequence) {
            let lhood_v = self.likelihood_v(v, delv);
            // drop that specific recombination event if the likelihood is too low
            if lhood_v < inference_params.min_likelihood {
                continue;
            }
            for (j, delj, d, deld5, deld3) in self.range_dj(sequence) {
                let e = EventVDJ {
                    v,
                    j,
                    d,
                    delv,
                    delj,
                    deld3,
                    deld5,
                };
                let lhood_dj = self.likelihood_dj(&e);
                let mut l_total = lhood_v * lhood_dj;
                // drop that specific recombination event if the likelihood is too low
                if l_total < inference_params.min_likelihood {
                    continue;
                }

                // extract both inserted sequences
                let (insvd, insdj) = sequence.get_insertions_vd_dj(EventVDJ {
                    v,
                    j,
                    d,
                    delv,
                    delj,
                    deld3,
                    deld5,
                });

                l_total *= self.insvd.likelihood(&insvd);
                l_total *= self.insdj.likelihood(&insdj);

                // drop that specific recombination event if the likelihood is too low
                if l_total < inference_params.min_likelihood {
                    continue;
                }
                println!("seq: {insvd} {l_total}");
                // Update everything with the new likelihood
                self.v.dirty_update(v.index, l_total);
                self.dj.dirty_update((d.index, j.index), l_total);
                self.delv.dirty_update((delv, v.index), l_total);
                self.delj.dirty_update((delj, j.index), l_total);
                self.deld.dirty_update((deld3, deld5, d.index), l_total);
                self.nb_insvd.dirty_update(insvd.len(), l_total);
                self.nb_insdj.dirty_update(insdj.len(), l_total);
                self.insvd.dirty_update(&insvd, l_total);
                self.insdj.dirty_update(&insdj, l_total);
                self.error.dirty_update(
                    j.nb_errors(delj) + v.nb_errors(delv) + d.nb_errors(deld3, deld5),
                    l_total,
                );
            }
        }
    }

    fn cleanup(&self) -> Result<FeaturesVDJ> {
        // Compute the new marginals for the next round
        Ok(FeaturesVDJ {
            v: self.v.cleanup()?,
            dj: self.dj.cleanup()?,
            delv: self.delv.cleanup()?,
            delj: self.delj.cleanup()?,
            deld: self.deld.cleanup()?,
            nb_insvd: self.nb_insvd.cleanup()?,
            nb_insdj: self.nb_insdj.cleanup()?,
            insvd: self.insvd.cleanup()?,
            insdj: self.insdj.cleanup()?,
            error: self.error.cleanup()?,
        })
    }
}
#[pymethods]
impl FeaturesVDJ {
    //fn most_likely(&self, n: usize) -> Result<Vec<Event>> {}
    #[staticmethod]
    fn average(features: Vec<FeaturesVDJ>) -> Result<FeaturesVDJ> {
        Ok(FeaturesVDJ {
            v: CategoricalFeature1::average(features.iter().map(|a| a.v.clone()))?,
            delv: CategoricalFeature1g1::average(features.iter().map(|a| a.delv.clone()))?,
            dj: CategoricalFeature2::average(features.iter().map(|a| a.dj.clone()))?,
            delj: CategoricalFeature1g1::average(features.iter().map(|a| a.delj.clone()))?,
            deld: CategoricalFeature2g1::average(features.iter().map(|a| a.deld.clone()))?,
            nb_insvd: CategoricalFeature1::average(features.iter().map(|a| a.nb_insvd.clone()))?,
            nb_insdj: CategoricalFeature1::average(features.iter().map(|a| a.nb_insdj.clone()))?,
            insvd: MarkovFeature::average(features.iter().map(|a| a.insvd.clone()))?,
            insdj: MarkovFeature::average(features.iter().map(|a| a.insdj.clone()))?,
            error: ErrorPoisson::average(features.iter().map(|a| a.error.clone()))?,
        })
    }
}

#[pyfunction]
pub fn infer_features(
    sequence: SequenceVDJ,
    model: ModelVDJ,
    inference_params: InferenceParameters,
) -> Result<FeaturesVDJ> {
    let mut feature = FeaturesVDJ::new(&model, &inference_params)?;
    feature.dirty_update(&sequence, &inference_params);
    feature.cleanup()?;
    Ok(feature)
}
