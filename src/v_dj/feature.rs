use crate::shared::data_structures::RangeArray1;
use crate::shared::{Feature, InferenceParameters, VJAlignment};
use crate::v_dj::Features;
use crate::vdj::feature::{AggregatedFeatureSpanD, FeatureDJ};
use crate::vdj::AggregatedFeatureStartJ;

use std::cmp;

// Contains the probability of the DJ gene group starting  position
pub struct AggregatedFeatureStartDAndJ {
    // range of possible values
    pub start_d5: i64,
    pub end_d5: i64,

    // Contains all the likelihood  P(d5, j)
    likelihood: RangeArray1,
    // Dirty likelihood, will be updated as we go through the inference
    dirty_likelihood: RangeArray1,

    // Store the associated J feature
    feature_j: AggregatedFeatureStartJ,

    // most likely start and end of the insertion sequence
    pub most_likely_d_end: i64,
    pub most_likely_j_start: i64,
    // most likely d_index
    pub most_likely_d_index: usize,
}

impl AggregatedFeatureStartDAndJ {
    pub fn new(
        j_alignment: &VJAlignment,
        feature_ds: &[AggregatedFeatureSpanD],
        feat_insdj: &FeatureDJ,
        feat: &Features,
        ip: &InferenceParameters,
    ) -> Option<AggregatedFeatureStartDAndJ> {
        let Some(feature_j) =
            AggregatedFeatureStartJ::new(j_alignment, &feat.delj, &feat.error, ip)
        else {
            return None;
        };

        // define the likelihood object with the right length
        let mut likelihood = RangeArray1::zeros((
            feature_ds.iter().map(|x| x.start_d5).min().unwrap(),
            feature_ds.iter().map(|x| x.end_d5).max().unwrap(),
        ));

        // for every parameters, iterate over delj / d / deld3 / deld5
        let mut total_likelihood = 0.;

        // to find the most likely events
        let mut most_likely_d_end = 0;
        let mut most_likely_d_index = 0;
        let mut most_likely_j_start = 0;
        let mut best_likelihood = 0.;

        // let (min_sj, max_sj) = (
        //     cmp::max(feature_j.start_j5, feat_insdj.min_sj()),
        //     cmp::min(feature_j.end_j5, feat_insdj.max_sj()),
        // );

        feature_ds.iter().for_each(|agg_d| {
            let ll_dj = feat.d.likelihood((agg_d.index, feature_j.index)); // P(D|J)
            feat_insdj.iter().for_each(|(d_end, j_start, ll_ins_dj)| {
                // j_start doesn't come from the iterator of feature_j
                // so we need to be careful here
                if (j_start >= feature_j.start_j5) && (j_start < feature_j.end_j5) {
                    let ll_ins_jd = feature_j.likelihood(j_start) * ll_ins_dj * ll_dj;

                    // iter_fixed_dend has the checks included
                    agg_d.iter_fixed_dend(d_end).for_each(|(d_start, ll_deld)| {
                        let ll = ll_ins_jd * ll_deld;
                        if ll > ip.min_likelihood {
                            if ll > best_likelihood {
                                most_likely_d_end = d_end;
                                most_likely_d_index = agg_d.index;
                                most_likely_j_start = j_start;
                                best_likelihood = ll;
                            }
                            *likelihood.get_mut(d_start) += ll;
                            total_likelihood += ll;
                        }
                    });
                }
            });
        });

        if total_likelihood == 0. {
            return None;
        }

        Some(AggregatedFeatureStartDAndJ {
            start_d5: likelihood.min,
            end_d5: likelihood.max,
            dirty_likelihood: RangeArray1::zeros(likelihood.dim()),
            likelihood,
            feature_j: feature_j,
            most_likely_d_end,
            most_likely_j_start,
            most_likely_d_index,
        })
    }

    pub fn j_start_seq(&self) -> usize {
        self.feature_j.start_seq
    }

    pub fn j_index(&self) -> usize {
        self.feature_j.index
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
        &mut self,
        j_alignment: &VJAlignment,
        feature_ds: &mut [AggregatedFeatureSpanD],
        feat_insdj: &mut FeatureDJ,
        feat: &mut Features,
        ip: &InferenceParameters,
    ) {
        let (min_sj, max_sj) = (
            cmp::max(self.feature_j.start_j5, feat_insdj.min_sj()),
            cmp::min(self.feature_j.end_j5, feat_insdj.max_sj()),
        );

        for agg_deld in feature_ds.iter_mut() {
            let (min_ed, max_ed) = (
                cmp::max(agg_deld.start_d3, feat_insdj.min_ed()),
                cmp::min(agg_deld.end_d3, feat_insdj.max_ed()),
            );

            let ll_dj = feat.d.likelihood((agg_deld.index, self.feature_j.index));
            for d_start in agg_deld.start_d5..agg_deld.end_d5 {
                let dirty_proba = self.dirty_likelihood.get(d_start);
                let ratio_old_new = dirty_proba / self.likelihood(d_start);
                for j_start in min_sj..max_sj {
                    let ll_j = self.feature_j.likelihood(j_start);
                    if ll_dj * ll_j < ip.min_likelihood {
                        continue;
                    }
                    for d_end in cmp::max(d_start - 1, min_ed)..cmp::min(max_ed, j_start + 1) {
                        let ll_deld = agg_deld.likelihood(d_start, d_end);

                        let ll_insdj = feat_insdj.likelihood(d_end, j_start);

                        let ll = ll_dj * ll_insdj * ll_j * ll_deld;

                        if ll > ip.min_likelihood {
                            let corrected_proba = ratio_old_new * ll;

                            if dirty_proba > 0. {
                                feat.d.dirty_update(
                                    (agg_deld.index, self.feature_j.index),
                                    corrected_proba,
                                );
                                if ip.infer_insertions {
                                    feat_insdj.dirty_update(d_end, j_start, corrected_proba);
                                }
                                self.feature_j.dirty_update(j_start, corrected_proba);
                                agg_deld.dirty_update(d_start, d_end, corrected_proba);
                            }
                        }
                    }
                }
            }
        }
        self.feature_j
            .disaggregate(j_alignment, &mut feat.delj, &mut feat.error, ip)
    }
}
