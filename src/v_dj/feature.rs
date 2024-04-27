use crate::shared::data_structures::RangeArray1;
use crate::shared::feature::Feature;
use crate::shared::{InferenceParameters, VJAlignment};
use crate::v_dj::Features;
use crate::vdj::feature::{AggregatedFeatureSpanD, FeatureDJ};

use std::cmp;

// Contains the probability of the DJ gene group starting  position
pub struct AggregatedFeatureStartDAndJ {
    pub index: usize,     // store the index of the J gene
    pub start_seq: usize, // store the start of the J gene

    // range of possible values
    pub start_d5: i64,
    pub end_d5: i64,

    // Contains all the likelihood  P(d5, j)
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
        jal: &VJAlignment,
        feature_ds: &Vec<AggregatedFeatureSpanD>,
        feat_insdj: &FeatureDJ,
        feat: &Features,
        ip: &InferenceParameters,
    ) -> Option<AggregatedFeatureStartDAndJ> {
        // define the likelihood object with the right length
        let mut likelihood = RangeArray1::zeros((
            feature_ds.iter().map(|x| x.start_d5).min().unwrap() as i64,
            feature_ds.iter().map(|x| x.end_d5).max().unwrap() as i64,
        ));

        // for every parameters, iterate over delj / d / deld3 / deld5
        let mut total_likelihood = 0.;

        // to find the most likely events
        let mut most_likely_d_end = 0;
        let mut most_likely_d_index = 0;
        let mut most_likely_j_start = 0;
        let mut best_likelihood = 0.;

        let mut likelihood_dstart = 0.;

        for d in feature_ds {
            for delj in 0..feat.delj.dim().0 {
                let (min_ed, max_ed) = (
                    cmp::max(d.start_d3, feat_insdj.min_ed()),
                    cmp::min(d.end_d3, feat_insdj.max_ed()),
                );
                let j_start = (jal.start_seq + delj) as i64;
                let ll_dj = feat.d.likelihood((d.index, jal.index));
                let ll_delj = feat.delj.likelihood((delj, jal.index));
                let ll_error_j = feat
                    .error
                    .likelihood((jal.nb_errors(delj), jal.length_with_deletion(delj)));

                if ll_dj * ll_delj * ll_error_j < ip.min_likelihood {
                    continue;
                }
                for d_start in d.start_d5..d.end_d5 {
                    for d_end in cmp::max(d_start - 1, min_ed)..max_ed {
                        let ll_d = d.likelihood(d_start, d_end);
                        let ll_insdj = feat_insdj.likelihood(d_end, j_start);
                        let ll = ll_dj * ll_insdj * ll_delj * ll_d * ll_error_j;

                        if ll > ip.min_likelihood {
                            if ll > best_likelihood {
                                most_likely_d_end = d_end;
                                most_likely_d_index = d.index;
                                most_likely_j_start = j_start;
                                best_likelihood = ll;
                            }
                            total_likelihood += ll;
                            likelihood_dstart += ll;
                        }
                    }
                    *likelihood.get_mut(d_start) += likelihood_dstart;
                    likelihood_dstart = 0.;
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
            index: jal.index,
            start_seq: jal.start_seq,
            most_likely_d_end,
            most_likely_j_start,
            most_likely_d_index,
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
        jal: &VJAlignment,
        feature_ds: &mut Vec<AggregatedFeatureSpanD>,
        feat_insdj: &mut FeatureDJ,
        feat: &mut Features,
        ip: &InferenceParameters,
    ) {
        for agg_deld in feature_ds.iter_mut() {
            for delj in 0..feat.delj.dim().0 {
                let (min_ed, max_ed) = (
                    cmp::max(agg_deld.start_d3, feat_insdj.min_ed()),
                    cmp::min(agg_deld.end_d3, feat_insdj.max_ed()),
                );
                let j_start = (jal.start_seq + delj) as i64;
                let ll_dj = feat.d.likelihood((agg_deld.index, jal.index));
                let ll_delj = feat.delj.likelihood((delj, jal.index));
                let ll_error_j = feat
                    .error
                    .likelihood((jal.nb_errors(delj), jal.length_with_deletion(delj)));

                if ll_dj * ll_delj * ll_error_j < ip.min_likelihood {
                    continue;
                }

                for d_start in agg_deld.start_d5..agg_deld.end_d5 {
                    for d_end in cmp::max(d_start - 1, min_ed)..max_ed {
                        let ll_deld = agg_deld.likelihood(d_start, d_end);
                        let ll_insdj = feat_insdj.likelihood(d_end, j_start);
                        let ll = ll_dj * ll_insdj * ll_delj * ll_deld * ll_error_j;

                        if ll > ip.min_likelihood {
                            let dirty_proba = self.dirty_likelihood.get(d_start);
                            let corrected_proba = dirty_proba * ll / self.likelihood(d_start);

                            if dirty_proba > 0. {
                                feat.d
                                    .dirty_update((agg_deld.index, jal.index), corrected_proba);
                                if ip.infer_insertions {
                                    feat_insdj.dirty_update(d_end, j_start, corrected_proba);
                                }
                                feat.delj.dirty_update((delj, jal.index), corrected_proba);
                                agg_deld.dirty_update(d_start, d_end, corrected_proba);
                                feat.error.dirty_update(
                                    (jal.nb_errors(delj), jal.length_with_deletion(delj)),
                                    corrected_proba,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
