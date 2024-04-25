use crate::shared::data_structures::RangeArray1;
use crate::shared::feature::Feature;
use crate::shared::{DAlignment, InferenceParameters, VJAlignment};
use crate::v_dj::Features;
use crate::vdj::feature::FeatureDJ;
use itertools::iproduct;
use std::collections::HashMap;

// Contains the probability of the DJ gene group starting  position
pub struct AggregatedFeatureStartDAndJ {
    pub index: usize,     // store the index of the J gene
    pub start_seq: usize, // store the start of the J gene

    // range of possible values
    pub start_d5: i64,
    pub end_d5: i64,

    // Contains all the likelihood  P(d5)
    likelihood: RangeArray1,
    // Dirty likelihood, will be updated as we go through the inference
    dirty_likelihood: RangeArray1,
    total_likelihood: f64,

    // most likely start and end of the insertion sequence
    pub most_likely_d_end: i64,
    pub most_likely_j_start: i64,
    // most likely d_index
    pub most_likely_d_index: usize,
}

impl AggregatedFeatureStartDAndJ {
    pub fn new(
        j: &VJAlignment,
        ds: &Vec<DAlignment>,
        feat_insdj: &FeatureDJ,
        feat: &Features,
        ip: &InferenceParameters,
    ) -> Option<AggregatedFeatureStartDAndJ> {
        // define the likelihood object with the right length
        let mut likelihood = RangeArray1::zeros((
            ds.iter().map(|x| x.pos).min().unwrap() as i64,
            ds.iter().map(|x| x.pos).max().unwrap() as i64 + feat.deld.dim().0 as i64,
        ));

        // for every parameters, iterate over delj / d / deld3 / deld5
        let mut total_likelihood = 0.;

        // hashmap to find the most likely d_start / j_end at the end
        let mut total_likelihood_d_end = HashMap::new();
        let mut total_likelihood_j_start = HashMap::new();
        let mut total_likelihood_d_index = HashMap::new();

        for d in ds {
            for (deld5, deld3) in iproduct!(0..feat.deld.dim().0, 0..feat.deld.dim().1) {
                for delj in 0..feat.delj.dim().0 {
                    let j_start = (j.start_seq + delj) as i64;
                    let d_start = (d.pos + deld5) as i64;
                    let d_end = (d.pos + d.len() - deld3) as i64;
                    if d_start > d_end {
                        continue;
                    }

                    let ll_dj = feat.d.likelihood((d.index, j.index));
                    let ll_insdj = feat_insdj.likelihood(d_end, j_start);
                    let ll_delj = feat.delj.likelihood((delj, j.index));
                    let ll_deld = feat.deld.likelihood((deld5, deld3, d.index));
                    let ll_error = feat.error.likelihood((
                        d.nb_errors(deld5, deld3) + j.nb_errors(delj),
                        d.length_with_deletion(deld5, deld3) + j.length_with_deletion(delj),
                    ));

                    let ll = ll_dj * ll_delj * ll_deld * ll_error * ll_insdj;
                    if ll > ip.min_likelihood {
                        *total_likelihood_d_end.entry(d_end).or_insert(0.) += ll;
                        *total_likelihood_j_start.entry(j_start).or_insert(0.) += ll;
                        *total_likelihood_d_index.entry(d.index).or_insert(0.) += ll;
                        *likelihood.get_mut(d_start) = ll;
                        total_likelihood += ll;
                    }
                }
            }
        }

        if total_likelihood == 0. {
            return None;
        }
        Some(AggregatedFeatureStartDAndJ {
            start_d5: likelihood.min,
            end_d5: likelihood.max,
            dirty_likelihood: RangeArray1::zeros(likelihood.dim()),
            likelihood,
            total_likelihood,
            index: j.index,
            start_seq: j.start_seq,
            // error handling is fine, if one of these is empty then total_likelihood is 0
            most_likely_d_end: total_likelihood_d_end
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _v)| *k)
                .unwrap(),
            most_likely_j_start: total_likelihood_j_start
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _v)| *k)
                .unwrap(),
            most_likely_d_index: total_likelihood_d_index
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _v)| *k)
                .unwrap(),
        })
    }

    pub fn likelihood(&self, sd: i64) -> f64 {
        self.likelihood.get(sd)
    }

    pub fn max_likelihood(&self) -> f64 {
        self.likelihood.max_value()
    }

    pub fn dirty_update(&mut self, sd: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut(sd) += likelihood;
    }

    pub fn disaggregate(
        &self,
        j: &VJAlignment,
        ds: &Vec<DAlignment>,
        feat_insdj: &mut FeatureDJ,
        feat: &mut Features,
        ip: &InferenceParameters,
    ) {
        for d in ds {
            for (deld5, deld3) in iproduct!(0..feat.deld.dim().0, 0..feat.deld.dim().1) {
                for delj in 0..feat.delj.dim().0 {
                    let j_start = (j.start_seq + delj) as i64;
                    let d_start = (d.pos + deld5) as i64;
                    let d_end = (d.pos + d.len() - deld3) as i64;

                    let nb_err = d.nb_errors(deld5, deld3) + j.nb_errors(delj);

                    let ll_dj = feat.d.likelihood((d.index, j.index));
                    let ll_insdj = feat_insdj.likelihood(d_end, j_start);
                    let ll_delj = feat.delj.likelihood((delj, j.index));
                    let ll_deld = feat.deld.likelihood((deld5, deld3, d.index));
                    let ll_error = feat.error.likelihood((
                        nb_err,
                        d.length_with_deletion(deld5, deld3) + j.length_with_deletion(delj),
                    ));

                    let ll = ll_dj * ll_delj * ll_deld * ll_error * ll_insdj;

                    if ll > ip.min_likelihood {
                        let proba_params_given_sd = ll / self.total_likelihood;
                        let likelihood = ll;
                        let dirty_proba = self.dirty_likelihood.get(d_start);
                        if dirty_proba > 0. {
                            let corrected_proba = dirty_proba * proba_params_given_sd;

                            feat.d.dirty_update((d.index, j.index), corrected_proba);

                            feat.deld
                                .dirty_update((deld5, deld3, d.index), corrected_proba);

                            feat.error.dirty_update(
                                (nb_err, d.length_with_deletion(deld5, deld3)),
                                corrected_proba,
                            );

                            if ip.infer_insertions {
                                feat_insdj.dirty_update(d_end, j_start, likelihood);
                            }

                            feat.delj.dirty_update((delj, j.index), corrected_proba);

                            feat.error.dirty_update(
                                (
                                    j.nb_errors(delj) + nb_err,
                                    j.length_with_deletion(delj)
                                        + d.length_with_deletion(deld5, deld3),
                                ),
                                corrected_proba,
                            )
                        }
                    }
                }
            }
        }
    }
}
