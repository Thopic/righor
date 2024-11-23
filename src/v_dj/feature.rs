use crate::shared::data_structures::RangeArray1;
use crate::shared::likelihood::{Likelihood, Likelihood1DContainer, LikelihoodType};
use crate::shared::sequence::SequenceType;

use crate::shared::InfEvent;
use crate::shared::{Feature, InferenceParameters, VJAlignment};
use crate::v_dj::Features;
use crate::vdj::feature::{AggregatedFeatureSpanD, FeatureDJ};
use crate::vdj::AggregatedFeatureStartJ;

use std::cmp;

// Contains the probability of the DJ gene group starting  position
#[derive(Debug)]
pub struct AggregatedFeatureStartDAndJ {
    // range of possible values
    pub start_d5: i64,
    pub end_d5: i64,

    // Contains all the likelihood  P(d5, j)
    likelihood: Likelihood1DContainer,
    // Dirty likelihood, will be updated as we go through the inference
    dirty_likelihood: RangeArray1,

    // Store the associated J feature
    feature_j: AggregatedFeatureStartJ,
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
        let mut likelihoods = Likelihood1DContainer::zeros(
            feature_ds.iter().map(|x| x.start_d5).min().unwrap(),
            feature_ds.iter().map(|x| x.end_d5).max().unwrap(),
            j_alignment.sequence_type,
        );

        let mut total_likelihood = Likelihood::zero_from_type(match j_alignment.sequence_type {
            SequenceType::Dna => LikelihoodType::Scalar,
            SequenceType::Protein => LikelihoodType::Vector,
        });

        // for every parameters, iterate over delj / d / deld3 / deld5
        // to find the most likely events
        feature_ds.iter().for_each(|agg_d| {
            let ll_dj = feat.d.likelihood((agg_d.index, feature_j.index)); // P(D|J)
            feat_insdj
                .iter()
                .for_each(|(last_nuc, d_end, j_start, ll_ins_dj)| {
                    // j_start needs to be compatible with feature_j
                    if (j_start >= feature_j.start_j5)
                        && (j_start < feature_j.end_j5)
                        && (last_nuc as usize
                            == j_alignment
                                .get_first_nucleotide((j_start - feature_j.start_j5) as usize))
                    {
                        let ll_d_and_j = ll_ins_dj * feature_j.likelihood(j_start);
                        // iter_fixed_dend iter only on valid values of d_start
                        agg_d.iter_fixed_dend(d_end).for_each(|(d_start, ll_deld)| {
                            let ll = ll_deld * ll_d_and_j.clone() * ll_dj;
                            if ll.max() > ip.min_likelihood {
                                likelihoods.add_to(d_start, ll.clone());
                                total_likelihood += ll;
                            }
                        });
                    }
                });
        });

        if total_likelihood.is_zero() {
            return None;
        }

        Some(AggregatedFeatureStartDAndJ {
            start_d5: likelihoods.min(),
            end_d5: likelihoods.max(),
            dirty_likelihood: RangeArray1::zeros(likelihoods.dim()),
            likelihood: likelihoods,
            feature_j,
        })
    }

    pub fn j_start_seq(&self) -> i64 {
        self.feature_j.start_seq
    }

    pub fn j_index(&self) -> usize {
        self.feature_j.index
    }

    pub fn likelihood(&self, sd: i64) -> Likelihood {
        self.likelihood.get(sd)
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
        event: &mut Option<InfEvent>,
        ip: &InferenceParameters,
    ) {
        // disaggregate should only work with the scalar version

        let (min_sj, max_sj) = (
            cmp::max(self.feature_j.start_j5, feat_insdj.min_sj()),
            cmp::min(self.feature_j.end_j5, feat_insdj.max_sj()),
        );

        let mut best_likelihood = 0.;
        for agg_deld in feature_ds.iter_mut() {
            let (min_ed, max_ed) = (
                cmp::max(agg_deld.start_d3, feat_insdj.min_ed()),
                cmp::min(agg_deld.end_d3, feat_insdj.max_ed()),
            );

            let ll_dj = feat.d.likelihood((agg_deld.index, self.feature_j.index));

            for d_start in agg_deld.start_d5..agg_deld.end_d5 {
                let dirty_proba = if ip.infer_features {
                    self.dirty_likelihood.get(d_start)
                } else {
                    0.
                };
                if dirty_proba == 0. && ip.infer_features {
                    continue;
                }
                let ratio_old_new = dirty_proba / self.likelihood(d_start).max();
                for j_start in min_sj..max_sj {
                    let ll_j = self.feature_j.likelihood(j_start);
                    if (ll_dj * ll_j.clone()).max() < ip.min_likelihood {
                        continue;
                    }
                    for d_end in cmp::max(d_start - 1, min_ed)..cmp::min(max_ed, j_start + 1) {
                        let ll_deld = agg_deld.likelihood(d_start, d_end);

                        let ll_insdj = feat_insdj.likelihood(
                            d_end,
                            j_start,
                            j_alignment
                                .get_first_nucleotide((j_start - self.feature_j.start_j5) as usize),
                        );

                        let ll = ll_dj * ll_deld * ll_insdj * ll_j.clone();

                        if ll.max() > ip.min_likelihood {
                            let corrected_proba = ratio_old_new * ll.max();
                            if ip.infer_features {
                                feat.d.dirty_update(
                                    (agg_deld.index, self.feature_j.index),
                                    corrected_proba,
                                );

                                feat_insdj.dirty_update(
                                    d_end,
                                    j_start,
                                    j_alignment.get_first_nucleotide(
                                        (j_start - self.feature_j.start_j5) as usize,
                                    ),
                                    corrected_proba,
                                );

                                self.feature_j.dirty_update(j_start, corrected_proba);
                                agg_deld.dirty_update(d_start, d_end, corrected_proba);
                            }

                            if ip.store_best_event && ll.max() > best_likelihood {
                                if let Some(ev) = event {
                                    if ev.start_d == d_start && ev.j_index == j_alignment.index {
                                        ev.d_index = agg_deld.index;
                                        ev.end_d = d_end;
                                        ev.start_j = j_start;
                                        best_likelihood = ll.max();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if ip.store_best_event {
            if let Some(ev) = event {
                ev.likelihood *= best_likelihood / self.likelihood(ev.start_d).max();
            }
        }

        self.feature_j
            .disaggregate(j_alignment, &mut feat.delj, &mut feat.error, ip)
    }
}
