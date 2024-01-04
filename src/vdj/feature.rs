use crate::sequence::utils::difference_as_i64;
use crate::shared::feature::Feature;
use crate::shared::utils::{RangeArray1, RangeArray2, RangeArray3};
use crate::shared::InferenceParameters;
use crate::vdj::{Features, Sequence};
use itertools::iproduct;
use ndarray::Array1;

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

    // Contains all the likelihood  P(startJ | J)
    log_likelihood_j: RangeArray2,
    // Contains all the likelihood  P(startD, endD | J)
    log_likelihood_d: RangeArray3,
    // Dirty likelihood, will be updated as we go through the inference
    dirty_likelihood_j: RangeArray2,
    dirty_likelihood_d: RangeArray3,
    total_likelihood_d: f64,
    total_likelihood_j: f64,
    pub nb_j_alignments: usize,
}

impl AggregatedFeatureDJ {
    pub fn new(
        sequence: &Sequence,
        feat: &Features,
        ip: &InferenceParameters,
    ) -> AggregatedFeatureDJ {
        let mut total_likelihood_j = 0.;
        let mut total_likelihood_d = 0.;
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

        // pre-compute the delj likelihood to
        let nb_j_al = sequence.j_genes.len();
        let mut log_likelihood_j = RangeArray2::zeros(((min.2, 0), (max.2 + 1, nb_j_al as i64)));
        let mut log_likelihood_d =
            RangeArray3::zeros(((min.0, min.1, 0), (max.0 + 1, max.1 + 1, nb_j_al as i64)));

        // compute P(J) = Σ_D P(D, J)
        let mut proba_j = Array1::<f64>::zeros(nb_j_al);
        for (idx_j, j) in sequence.j_genes.iter().enumerate() {
            for d in sequence.d_genes.iter() {
                proba_j[idx_j] += feat.dj.log_likelihood((d.index, j.index)).exp2();
            }
        }
        let log_proba_j = proba_j.mapv(|x| x.log2());

        for (idx_j, j) in sequence.j_genes.iter().enumerate() {
            // First deal with the j likelihood
            for delj in 0..feat.delj.dim().0 {
                let j_start = (j.start_seq + delj) as i64;
                let ll_j = log_proba_j[idx_j]
                    + feat.delj.log_likelihood((delj, j.index))
                    + feat
                        .error
                        .log_likelihood((j.nb_errors(delj), j.length_with_deletion(delj)));

                if ll_j > ip.min_log_likelihood {
                    let likelihood = ll_j.exp2();
                    *log_likelihood_j.get_mut((j_start, idx_j as i64)) += likelihood;
                    total_likelihood_j += likelihood;
                }
            }

            // Now with startD and end D
            for d in sequence.d_genes.iter() {
                for (deld5, deld3) in iproduct!(0..feat.deld.dim().1, 0..feat.deld.dim().0) {
                    let d_start = (d.pos + deld5) as i64;
                    let d_end = (d.pos + d.len() - deld3) as i64;
                    if d_start > d_end {
                        continue;
                    }
                    let ll_d = feat.deld.log_likelihood((deld3, deld5, d.index))
                        + feat.error.log_likelihood((
                            d.nb_errors(deld5, deld3),
                            d.length_with_deletion(deld5, deld3),
                        ));
                    if ll_d > ip.min_log_likelihood {
                        let likelihood = ll_d.exp2();
                        *log_likelihood_d.get_mut((d_start, d_end, idx_j as i64)) += likelihood;
                        total_likelihood_d += likelihood;
                    }
                }
            }
        }

        log_likelihood_d.mut_map(|x| x.log2());
        log_likelihood_j.mut_map(|x| x.log2());

        AggregatedFeatureDJ {
            start_d5: log_likelihood_d.min.0,
            end_d5: log_likelihood_d.max.0,
            start_d3: log_likelihood_d.min.1,
            end_d3: log_likelihood_d.max.1,
            start_j5: log_likelihood_j.min.0,
            end_j5: log_likelihood_j.max.0,
            dirty_likelihood_d: RangeArray3::zeros(log_likelihood_d.dim()),
            dirty_likelihood_j: RangeArray2::zeros(log_likelihood_j.dim()),
            log_likelihood_d,
            log_likelihood_j,
            total_likelihood_d,
            total_likelihood_j,
            nb_j_alignments: nb_j_al,
        }
    }

    pub fn log_likelihood(&self, sd: i64, ed: i64, sj: i64, j_idx: usize) -> f64 {
        self.log_likelihood_d.get((sd, ed, j_idx as i64))
            + self.log_likelihood_j.get((sj, j_idx as i64))
    }

    pub fn dirty_update(&mut self, sd: i64, ed: i64, sj: i64, j_idx: usize, likelihood: f64) {
        // Just update the marginals and hope for the best
        *self.dirty_likelihood_d.get_mut((sd, ed, j_idx as i64)) += likelihood;
        *self.dirty_likelihood_j.get_mut((sj, j_idx as i64)) += likelihood;
    }

    pub fn cleanup(&self, sequence: &Sequence, feat: &mut Features, ip: &InferenceParameters) {
        // compute P(J) = Σ_D P(D, J)
        let mut proba_j = Array1::zeros(self.nb_j_alignments);
        for (idx_j, j) in sequence.j_genes.iter().enumerate() {
            for d in sequence.d_genes.iter() {
                proba_j[idx_j] += feat.dj.log_likelihood((d.index, j.index)).exp2();
            }
        }
        let log_proba_j = proba_j.mapv(|x: f64| x.log2());

        for (j_idx, j) in sequence.j_genes.iter().enumerate() {
            // First deal with the j likelihood
            for delj in 0..feat.delj.dim().0 {
                let j_start = (j.start_seq + delj) as i64;
                let ll_j = log_proba_j[j_idx]
                    + feat.delj.log_likelihood((delj, j.index))
                    + feat
                        .error
                        .log_likelihood((j.nb_errors(delj), j.length_with_deletion(delj)));

                if ll_j > ip.min_log_likelihood {
                    let likelihood = ll_j.exp2();
                    let dirty_proba = self.dirty_likelihood_j.get((j_start, j_idx as i64));
                    if dirty_proba > 0. {
                        for d in sequence.d_genes.iter() {
                            feat.dj.dirty_update(
                                (d.index, j.index),
                                dirty_proba * likelihood
                                    / (self.total_likelihood_j * sequence.d_genes.len() as f64),
                            );
                        }

                        feat.delj.dirty_update(
                            (delj, j.index),
                            dirty_proba * likelihood / self.total_likelihood_j,
                        );

                        feat.error.dirty_update(
                            (j.nb_errors(delj), j.length_with_deletion(delj)),
                            dirty_proba * likelihood / self.total_likelihood_j,
                        );
                    }
                }
            }

            // Now with startD and end D
            for d in sequence.d_genes.iter() {
                for (deld5, deld3) in iproduct!(0..feat.deld.dim().1, 0..feat.deld.dim().0) {
                    let d_start = (d.pos + deld5) as i64;
                    let d_end = (d.pos + d.len() - deld3) as i64;
                    if d_start > d_end {
                        continue;
                    }
                    let ll_d = feat.deld.log_likelihood((deld3, deld5, d.index))
                        + feat.error.log_likelihood((
                            d.nb_errors(deld5, deld3),
                            d.length_with_deletion(deld5, deld3),
                        ));

                    if ll_d > ip.min_log_likelihood {
                        let likelihood = ll_d.exp2();
                        let dirty_proba =
                            self.dirty_likelihood_d.get((d_start, d_end, j_idx as i64));
                        if dirty_proba > 0. {
                            feat.dj.dirty_update(
                                (d.index, j.index),
                                dirty_proba * likelihood / self.total_likelihood_d,
                            );

                            feat.deld.dirty_update(
                                (deld3, deld5, d.index),
                                dirty_proba * likelihood / self.total_likelihood_d,
                            );

                            feat.error.dirty_update(
                                (
                                    d.nb_errors(deld5, deld3),
                                    d.length_with_deletion(deld5, deld3),
                                ),
                                dirty_proba * likelihood / self.total_likelihood_d,
                            );
                        }
                    }
                }
            }
        }
    }
}
