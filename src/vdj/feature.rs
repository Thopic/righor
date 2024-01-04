use crate::sequence::utils::difference_as_i64;
use crate::shared::feature::Feature;
use crate::shared::utils::{RangeArray1, RangeArray3};
use crate::shared::InferenceParameters;
use crate::vdj::{Features, Sequence};
use itertools::iproduct;

/// Contains the probability of the V gene ending at position e_v
/// For all reasonnable e_v
pub struct AggregatedFeatureEndV {
    // deal with the range of possible values for endV
    pub start_v3: i64,
    pub end_v3: i64,

    // Contains all the log-likelihood
    log_likelihood: RangeArray1,
    // sum of all the likelihood on v/delv
    total_likelihood: f64,

    // Dirty likelihood (will be updated as we go through the inference)
    dirty_likelihood: RangeArray1,
}

impl AggregatedFeatureEndV {
    pub fn new(
        sequence: &Sequence,
        feat: &Features,
        ip: &InferenceParameters,
    ) -> AggregatedFeatureEndV {
        let mut likelihood = Vec::new();
        let mut total_likelihood = 0.;
        for v in sequence.v_genes.iter() {
            for delv in 0..feat.delv.dim().0 {
                let v_end = difference_as_i64(v.end_seq, delv);
                let ll = feat.v.log_likelihood(v.index)
                    + feat.delv.log_likelihood((delv, v.index))
                    + feat
                        .error
                        .log_likelihood((v.nb_errors(delv), v.length_with_deletion(delv)));
                if ll > ip.min_log_likelihood {
                    likelihood.push((v_end, ll.exp2()));
                    total_likelihood += ll.exp2();
                }
            }
        }
        let mut log_likelihood = RangeArray1::new(&likelihood);
        log_likelihood.mut_map(|x| x.log2());

        AggregatedFeatureEndV {
            start_v3: log_likelihood.min,
            end_v3: log_likelihood.max,
            dirty_likelihood: RangeArray1::zeros(log_likelihood.dim()),
            log_likelihood,
            total_likelihood,
        }
    }

    pub fn log_likelihood(&self, ev: i64) -> f64 {
        self.log_likelihood.get(ev)
    }

    pub fn dirty_update(&mut self, ev: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut(ev) += likelihood;
    }

    pub fn cleanup(&self, sequence: &Sequence, feat: &mut Features, ip: &InferenceParameters) {
        for v in sequence.v_genes.iter() {
            for delv in 0..feat.delv.dim().0 {
                let v_end = difference_as_i64(v.end_seq, delv);
                let ll = feat.v.log_likelihood(v.index)
                    + feat.delv.log_likelihood((delv, v.index))
                    + feat
                        .error
                        .log_likelihood((v.nb_errors(delv), v.length_with_deletion(delv)));

                if ll > ip.min_log_likelihood {
                    let likelihood = ll.exp2();
                    let dirty_proba = self.dirty_likelihood.get(v_end);
                    if dirty_proba > 0. {
                        feat.v.dirty_update(
                            v.index,
                            dirty_proba * likelihood / self.total_likelihood,
                        );
                        feat.delv.dirty_update(
                            (delv, v.index),
                            dirty_proba * likelihood / self.total_likelihood,
                        );

                        feat.error.dirty_update(
                            (v.nb_errors(delv), v.length_with_deletion(delv)),
                            dirty_proba * likelihood / self.total_likelihood,
                        )
                    }
                }
            }
        }
    }
}

/// Contains the probability of the D and J genes starting and ending
/// at s_D, e_D, s_J respectively, this for a specific Sequence object
pub struct AggregatedFeatureDJ {
    // range of possible values
    pub start_d5: i64,
    pub end_d5: i64,
    pub start_d3: i64,
    pub end_d3: i64,
    pub start_j5: i64,
    pub end_j5: i64,

    // Contains all the log-likelihood
    log_likelihood: RangeArray3,
    // Dirty likelihood, will be updated as we go through the inference
    dirty_likelihood: RangeArray3,
    total_likelihood: f64,
}

impl AggregatedFeatureDJ {
    pub fn new(
        sequence: &Sequence,
        feat: &Features,
        ip: &InferenceParameters,
    ) -> AggregatedFeatureDJ {
        let mut total_likelihood = 0.;
        let min = (
            sequence.d_genes.iter().map(|x| x.pos).min().unwrap() as i64,
            sequence.d_genes.iter().map(|x| x.pos).min().unwrap() as i64
                + sequence.d_genes.iter().map(|x| x.len()).min().unwrap() as i64
                - feat.deld.dim().0 as i64,
            sequence.j_genes.iter().map(|x| x.start_seq).min().unwrap() as i64,
        );

        let max = (
            (sequence.d_genes.iter().map(|x| x.pos).max().unwrap() + feat.deld.dim().1) as i64,
            sequence.d_genes.iter().map(|x| x.pos).max().unwrap() as i64
                + sequence.d_genes.iter().map(|x| x.len()).max().unwrap() as i64,
            (sequence.j_genes.iter().map(|x| x.start_seq).max().unwrap() + feat.delj.dim().0)
                as i64,
        );

        let mut log_likelihood = RangeArray3::zeros((min, (max.0 + 1, max.1 + 1, max.2 + 1)));

        for (j, d) in iproduct!(sequence.j_genes.iter(), sequence.d_genes.iter()) {
            for delj in 0..feat.delj.dim().0 {
                let j_start = (j.start_seq + delj) as i64;
                let ll_pre_deld = feat.dj.log_likelihood((d.index, j.index))
                    + feat.delj.log_likelihood((delj, j.index))
                    + feat
                        .error
                        .log_likelihood((j.nb_errors(delj), j.length_with_deletion(delj)));

                for (deld5, deld3) in iproduct!(0..feat.deld.dim().1, 0..feat.deld.dim().0) {
                    let d_start = (d.pos + deld5) as i64;
                    let d_end = (d.pos + d.len() - deld3) as i64;

                    if (d_start > d_end) || (d_end > j_start) {
                        continue;
                    }
                    let ll = ll_pre_deld
                        + feat.deld.log_likelihood((deld3, deld5, d.index))
                        + feat.error.log_likelihood((
                            d.nb_errors(deld5, deld3),
                            d.length_with_deletion(deld5, deld3),
                        ));
                    if ll > ip.min_log_likelihood {
                        let likelihood = ll.exp2();
                        *log_likelihood.get_mut((d_start, d_end, j_start)) += likelihood;
                        total_likelihood += likelihood;
                    }
                }
            }
        }

        log_likelihood.mut_map(|x| x.log2());

        AggregatedFeatureDJ {
            start_d5: log_likelihood.min.0,
            end_d5: log_likelihood.max.0,
            start_d3: log_likelihood.min.1,
            end_d3: log_likelihood.max.1,
            start_j5: log_likelihood.min.2,
            end_j5: log_likelihood.max.2,
            dirty_likelihood: RangeArray3::zeros(log_likelihood.dim()),
            log_likelihood,
            total_likelihood,
        }
    }

    pub fn log_likelihood(&self, sd: i64, ed: i64, sj: i64) -> f64 {
        self.log_likelihood.get((sd, ed, sj))
    }

    pub fn dirty_update(&mut self, sd: i64, ed: i64, sj: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut((sd, ed, sj)) += likelihood;
    }

    pub fn cleanup(&self, sequence: &Sequence, feat: &mut Features, ip: &InferenceParameters) {
        for (j, d) in iproduct!(sequence.j_genes.iter(), sequence.d_genes.iter()) {
            for delj in 0..feat.delj.dim().0 {
                let j_start = (j.start_seq + delj) as i64;
                let ll_pre_deld = feat.dj.log_likelihood((d.index, j.index))
                    + feat.delj.log_likelihood((delj, j.index))
                    + feat
                        .error
                        .log_likelihood((j.nb_errors(delj), j.length_with_deletion(delj)));

                for (deld5, deld3) in iproduct!(0..feat.deld.dim().1, 0..feat.deld.dim().0) {
                    let d_start = (d.pos + deld5) as i64;
                    let d_end = (d.pos + d.len() - deld3) as i64;
                    if (d_start > d_end) || (d_end > j_start) {
                        continue;
                    }
                    let ll = ll_pre_deld
                        + feat.deld.log_likelihood((deld3, deld5, d.index))
                        + feat.error.log_likelihood((
                            d.nb_errors(deld5, deld3),
                            d.length_with_deletion(deld5, deld3),
                        ));
                    if ll > ip.min_log_likelihood {
                        let dirty_proba = self.dirty_likelihood.get((d_start, d_end, j_start));
                        if dirty_proba > 0. {
                            let likelihood = ll.exp2();
                            feat.dj.dirty_update(
                                (d.index, j.index),
                                dirty_proba * likelihood / self.total_likelihood,
                            );
                            feat.delj.dirty_update(
                                (delj, j.index),
                                dirty_proba * likelihood / self.total_likelihood,
                            );

                            feat.deld.dirty_update(
                                (deld3, deld5, d.index),
                                dirty_proba * likelihood / self.total_likelihood,
                            );

                            feat.error.dirty_update(
                                (
                                    j.nb_errors(delj) + d.nb_errors(deld5, deld3),
                                    j.length_with_deletion(delj)
                                        + d.length_with_deletion(deld5, deld3),
                                ),
                                dirty_proba * likelihood / self.total_likelihood,
                            )
                        }
                    }
                }
            }
        }
    }
}
