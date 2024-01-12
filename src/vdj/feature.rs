use crate::sequence::utils::{difference_as_i64, Dna};
use crate::sequence::{DAlignment, VJAlignment};
use crate::shared::feature::Feature;
use crate::shared::utils::{RangeArray1, RangeArray2};
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

pub struct AggregatedFeatureStartJ {
    // deal with the range of possible values for startJ
    pub start_j5: i64,
    pub end_j5: i64,

    // Contains all the log-likelihood
    log_likelihood: RangeArray1,
    // sum of all the likelihood on j/delj
    total_likelihood: f64,

    // Dirty likelihood (will be updated as we go through the inference)
    dirty_likelihood: RangeArray1,
}

// Contains the probability of the D gene starting and ending position
pub struct AggregatedFeatureSpanD {
    // range of possible values
    pub start_d5: i64,
    pub end_d5: i64,
    pub start_d3: i64,
    pub end_d3: i64,

    // Contains all the likelihood  P(startD, endD | D)
    log_likelihood: RangeArray2,
    // Dirty likelihood, will be updated as we go through the inference
    dirty_likelihood: RangeArray2,
    total_likelihood: f64,
}

impl AggregatedFeatureEndV {
    pub fn new(
        v: &VJAlignment,
        feat: &Features,
        ip: &InferenceParameters,
    ) -> Option<AggregatedFeatureEndV> {
        let mut log_likelihood = RangeArray1::zeros((
            difference_as_i64(v.end_seq, feat.delv.dim().0) + 1,
            v.end_seq as i64 + 1,
        ));
        let mut total_likelihood = 0.;
        for delv in 0..feat.delv.dim().0 {
            let v_end = difference_as_i64(v.end_seq, delv);
            let ll = feat.v.log_likelihood(v.index)
                + feat.delv.log_likelihood((delv, v.index))
                + feat
                    .error
                    .log_likelihood((v.nb_errors(delv), v.length_with_deletion(delv)));
            if ll > ip.min_log_likelihood {
                *log_likelihood.get_mut(v_end) = ll;
                total_likelihood += ll.exp2();
            }
        }

        if total_likelihood == 0. {
            return None;
        }

        Some(AggregatedFeatureEndV {
            start_v3: log_likelihood.min,
            end_v3: log_likelihood.max,
            dirty_likelihood: RangeArray1::zeros(log_likelihood.dim()),
            log_likelihood,
            total_likelihood,
        })
    }

    pub fn log_likelihood(&self, ev: i64) -> f64 {
        self.log_likelihood.get(ev)
    }

    pub fn dirty_update(&mut self, ev: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut(ev) += likelihood;
    }

    pub fn disaggregate(&self, v: &VJAlignment, feat: &mut Features, ip: &InferenceParameters) {
        for delv in 0..feat.delv.dim().0 {
            let v_end = difference_as_i64(v.end_seq, delv);
            let ll = feat.v.log_likelihood(v.index)
                + feat.delv.log_likelihood((delv, v.index))
                + feat
                    .error
                    .log_likelihood((v.nb_errors(delv), v.length_with_deletion(delv)));

            if ll > ip.min_log_likelihood {
                let proba_params_given_ev = ll.exp2() / self.total_likelihood; // P(parameters|ev)
                let dirty_proba = self.dirty_likelihood.get(v_end); // P(ev)
                if dirty_proba > 0. {
                    feat.v
                        .dirty_update(v.index, dirty_proba * proba_params_given_ev);
                    feat.delv
                        .dirty_update((delv, v.index), dirty_proba * proba_params_given_ev);
                    feat.error.dirty_update(
                        (v.nb_errors(delv), v.length_with_deletion(delv)),
                        dirty_proba * proba_params_given_ev,
                    );
                }
            }
        }
    }
}

impl AggregatedFeatureStartJ {
    pub fn new(
        v: &VJAlignment,
        j: &VJAlignment,
        feat: &Features,
        ip: &InferenceParameters,
    ) -> Option<AggregatedFeatureStartJ> {
        let mut log_likelihood = RangeArray1::zeros((
            j.start_seq as i64,
            (j.start_seq + feat.delj.dim().0) as i64 - 1,
        ));
        let mut total_likelihood = 0.;
        for delj in 0..feat.delj.dim().0 {
            let j_start = (j.start_seq + delj) as i64;
            let ll = feat.j.log_likelihood((j.index, v.index))
                + feat.delj.log_likelihood((delj, j.index))
                + feat
                    .error
                    .log_likelihood((j.nb_errors(delj), j.length_with_deletion(delj)));
            if ll > ip.min_log_likelihood {
                *log_likelihood.get_mut(j_start) = ll;
                total_likelihood += ll.exp2();
            }
        }

        if total_likelihood == 0. {
            return None;
        }
        Some(AggregatedFeatureStartJ {
            start_j5: log_likelihood.min,
            end_j5: log_likelihood.max,
            dirty_likelihood: RangeArray1::zeros(log_likelihood.dim()),
            log_likelihood,
            total_likelihood,
        })
    }

    pub fn log_likelihood(&self, sj: i64) -> f64 {
        self.log_likelihood.get(sj)
    }

    pub fn dirty_update(&mut self, sj: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut(sj) += likelihood;
    }

    pub fn disaggregate(
        &self,
        v: &VJAlignment,
        j: &VJAlignment,
        feat: &mut Features,
        ip: &InferenceParameters,
    ) {
        for delj in 0..feat.delj.dim().0 {
            let j_start = (j.start_seq + delj) as i64;
            let ll = feat.j.log_likelihood((j.index, v.index))
                + feat.delj.log_likelihood((delj, j.index))
                + feat
                    .error
                    .log_likelihood((j.nb_errors(delj), j.length_with_deletion(delj)));

            if ll > ip.min_log_likelihood {
                let proba_params_given_sj = ll.exp2() / self.total_likelihood; // P(delj, j, errors ...|sj)
                let likelihood = ll.exp2();
                let dirty_proba = self.dirty_likelihood.get(j_start);
                if dirty_proba > 0. {
                    feat.j
                        .dirty_update((j.index, v.index), dirty_proba * proba_params_given_sj);
                    feat.delj
                        .dirty_update((delj, j.index), dirty_proba * proba_params_given_sj);

                    feat.error.dirty_update(
                        (j.nb_errors(delj), j.length_with_deletion(delj)),
                        dirty_proba * likelihood / self.total_likelihood,
                    )
                }
            }
        }
    }
}

impl AggregatedFeatureSpanD {
    pub fn new(
        v: &VJAlignment,
        ds: &Vec<DAlignment>,
        j: &VJAlignment,
        feat: &Features,
        ip: &InferenceParameters,
    ) -> AggregatedFeatureSpanD {
        let mut total_likelihood = 0.;
        let mut log_likelihoods = RangeArray2::zeros((
            (
                // min start, min end
                ds.iter().map(|x| x.pos).min().unwrap() as i64,
                ds.iter().map(|x| x.pos + x.len()).min().unwrap() as i64 - feat.deld.dim().0 as i64
                    + 1,
            ),
            (
                // max start, max end
                ds.iter().map(|x| x.pos).max().unwrap() as i64 + feat.deld.dim().1 as i64,
                ds.iter().map(|x| x.pos + x.len()).max().unwrap() as i64 + 1,
            ),
        ));

        for d in ds {
            for (deld5, deld3) in iproduct!(0..feat.deld.dim().1, 0..feat.deld.dim().0) {
                let d_start = (d.pos + deld5) as i64;
                let d_end = (d.pos + d.len() - deld3) as i64;
                if d_start > d_end {
                    continue;
                }
                let ll_d = feat.d.log_likelihood((d.index, v.index, j.index))
                    + feat.deld.log_likelihood((deld3, deld5, d.index))
                    + feat.error.log_likelihood((
                        d.nb_errors(deld5, deld3),
                        d.length_with_deletion(deld5, deld3),
                    ));
                if ll_d > ip.min_log_likelihood {
                    let likelihood = ll_d.exp2();
                    *log_likelihoods.get_mut((d_start, d_end)) += likelihood;
                    total_likelihood += likelihood;
                }
            }
        }

        log_likelihoods.mut_map(|x| x.log2());

        AggregatedFeatureSpanD {
            start_d5: log_likelihoods.min.0,
            end_d5: log_likelihoods.max.0,
            start_d3: log_likelihoods.min.1,
            end_d3: log_likelihoods.max.1,
            dirty_likelihood: RangeArray2::zeros(log_likelihoods.dim()),
            log_likelihood: log_likelihoods,
            total_likelihood,
        }
    }

    pub fn log_likelihood(&self, sd: i64, ed: i64) -> f64 {
        self.log_likelihood.get((sd, ed))
    }

    pub fn dirty_update(&mut self, sd: i64, ed: i64, likelihood: f64) {
        // Just update the marginals and hope for the best
        *self.dirty_likelihood.get_mut((sd, ed)) += likelihood;
    }

    pub fn disaggregate(
        &self,
        v: &VJAlignment,
        ds: &Vec<DAlignment>,
        j: &VJAlignment,
        feat: &mut Features,
        ip: &InferenceParameters,
    ) {
        // Now with startD and end D
        for d in ds.iter() {
            for (deld5, deld3) in iproduct!(0..feat.deld.dim().1, 0..feat.deld.dim().0) {
                let d_start = (d.pos + deld5) as i64;
                let d_end = (d.pos + d.len() - deld3) as i64;
                if d_start > d_end {
                    continue;
                }
                let ll_d = feat.d.log_likelihood((d.index, v.index, j.index))
                    + feat.deld.log_likelihood((deld3, deld5, d.index))
                    + feat.error.log_likelihood((
                        d.nb_errors(deld5, deld3),
                        d.length_with_deletion(deld5, deld3),
                    ));

                if ll_d > ip.min_log_likelihood {
                    let proba_params_given_dspan = ll_d.exp2() / self.total_likelihood;
                    let dirty_proba = self.dirty_likelihood.get((d_start, d_end));
                    if dirty_proba > 0. {
                        feat.d.dirty_update(
                            (d.index, v.index, j.index),
                            dirty_proba * proba_params_given_dspan,
                        );

                        feat.deld.dirty_update(
                            (deld3, deld5, d.index),
                            dirty_proba * proba_params_given_dspan,
                        );

                        feat.error.dirty_update(
                            (
                                d.nb_errors(deld5, deld3),
                                d.length_with_deletion(deld5, deld3),
                            ),
                            dirty_proba * proba_params_given_dspan,
                        );
                    }
                }
            }
        }
    }
}

pub struct FeatureVD {
    log_likelihood: RangeArray2,
    dirty_likelihood: RangeArray2,
}

impl FeatureVD {
    pub fn new(sequence: &Sequence, feat: &Features, ip: &InferenceParameters) -> FeatureVD {
        let min_end_v = sequence.v_genes.iter().map(|x| x.end_seq).min().unwrap() as i64
            - feat.delv.dim().0 as i64
            + 1;
        let min_start_d = sequence.d_genes.iter().map(|x| x.pos).min().unwrap() as i64;
        let max_end_v = sequence.v_genes.iter().map(|x| x.end_seq).max().unwrap() as i64;
        let max_start_d = sequence.d_genes.iter().map(|x| x.pos).max().unwrap() as i64
            + feat.deld.dim().1 as i64
            - 1;

        let mut values = Vec::with_capacity(
            ((max_end_v + 1 - min_end_v) * (max_start_d + 1 - min_start_d)) as usize,
        );
        for ev in min_end_v..=max_end_v {
            for sd in min_start_d..=max_start_d {
                if sd >= ev && ((sd - ev) as usize) < feat.insvd.max_nb_insertions() {
                    let ins_vd_plus_first = sequence.get_subsequence(ev - 1, sd);
                    let ll = feat.insvd.log_likelihood(&ins_vd_plus_first);
                    if ll > ip.min_log_likelihood {
                        values.push(((ev, sd), ll));
                    }
                }
            }
        }

        let log_likelihood = RangeArray2::new(&values);
        FeatureVD {
            dirty_likelihood: RangeArray2::zeros(log_likelihood.dim()),
            log_likelihood,
        }
    }

    pub fn max_ev(&self) -> i64 {
        self.log_likelihood.max.0
    }

    pub fn min_ev(&self) -> i64 {
        self.log_likelihood.min.0
    }

    pub fn max_sd(&self) -> i64 {
        self.log_likelihood.max.1
    }

    pub fn min_sd(&self) -> i64 {
        self.log_likelihood.min.1
    }

    pub fn log_likelihood(&self, ev: i64, sd: i64) -> f64 {
        self.log_likelihood.get((ev, sd))
    }

    pub fn dirty_update(&mut self, ev: i64, sd: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut((ev, sd)) += likelihood;
    }

    pub fn disaggregate(&self, sequence: &Dna, feat: &mut Features, ip: &InferenceParameters) {
        for ev in self.log_likelihood.lower().0..self.log_likelihood.upper().0 {
            for sd in self.log_likelihood.lower().1..self.log_likelihood.upper().1 {
                if sd >= ev
                    && ((sd - ev) as usize) < feat.insvd.max_nb_insertions()
                    && self.log_likelihood(ev, sd) > ip.min_log_likelihood
                {
                    let ins_vd_plus_first = &sequence.extract_padded_subsequence(ev - 1, sd);
                    let ll = self.log_likelihood(ev, sd);
                    if ll > ip.min_log_likelihood {
                        feat.insvd
                            .dirty_update(ins_vd_plus_first, self.dirty_likelihood.get((ev, sd)))
                    }
                }
            }
        }
    }
}

pub struct FeatureDJ {
    log_likelihood: RangeArray2,
    dirty_likelihood: RangeArray2,
}

impl FeatureDJ {
    pub fn new(sequence: &Sequence, feat: &Features, ip: &InferenceParameters) -> FeatureDJ {
        let min_end_d = sequence
            .d_genes
            .iter()
            .map(|x| x.pos + x.len())
            .min()
            .unwrap() as i64
            - feat.deld.dim().0 as i64
            + 1;
        let min_start_j = sequence.j_genes.iter().map(|x| x.start_seq).min().unwrap() as i64;
        let max_end_d = sequence
            .d_genes
            .iter()
            .map(|x| x.pos + x.len())
            .max()
            .unwrap() as i64;
        let max_start_j = sequence.j_genes.iter().map(|x| x.start_seq).max().unwrap() as i64
            + feat.delj.dim().0 as i64
            - 1;

        let mut values = Vec::with_capacity(
            ((max_end_d + 1 - min_end_d) * (max_start_j + 1 - min_start_j)) as usize,
        );
        for ed in min_end_d..=max_end_d {
            for sj in min_start_j..=max_start_j {
                if sj >= ed && ((sj - ed) as usize) < feat.insdj.max_nb_insertions() {
                    // careful we need to reverse ins_dj for the inference
                    let mut ins_dj_plus_last = sequence.get_subsequence(ed, sj + 1);
                    ins_dj_plus_last.reverse();
                    let ll = feat.insdj.log_likelihood(&ins_dj_plus_last);
                    if ll > ip.min_log_likelihood {
                        values.push(((ed, sj), ll));
                    }
                }
            }
        }
        let log_likelihood = RangeArray2::new(&values);
        FeatureDJ {
            dirty_likelihood: RangeArray2::zeros(log_likelihood.dim()),
            log_likelihood,
        }
    }

    pub fn log_likelihood(&self, ed: i64, sj: i64) -> f64 {
        self.log_likelihood.get((ed, sj))
    }

    pub fn max_ed(&self) -> i64 {
        self.log_likelihood.max.0
    }

    pub fn min_ed(&self) -> i64 {
        self.log_likelihood.min.0
    }

    pub fn max_sj(&self) -> i64 {
        self.log_likelihood.max.1
    }

    pub fn min_sj(&self) -> i64 {
        self.log_likelihood.min.1
    }

    pub fn dirty_update(&mut self, ed: i64, sj: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut((ed, sj)) += likelihood;
    }

    pub fn disaggregate(&self, sequence: &Dna, feat: &mut Features, ip: &InferenceParameters) {
        for ed in self.log_likelihood.lower().0..self.log_likelihood.upper().0 {
            for sj in self.log_likelihood.lower().1..self.log_likelihood.upper().1 {
                if sj >= ed
                    && ((sj - ed) as usize) < feat.insdj.max_nb_insertions()
                    && self.log_likelihood(ed, sj) > ip.min_log_likelihood
                {
                    let mut ins_dj_plus_last = sequence.extract_padded_subsequence(ed, sj + 1);
                    ins_dj_plus_last.reverse();
                    feat.insdj
                        .dirty_update(&ins_dj_plus_last, self.dirty_likelihood.get((ed, sj)));
                }
            }
        }
    }
}
