use crate::shared::data_structures::{RangeArray1, RangeArray2};
use crate::shared::feature::Feature;
use crate::shared::utils::difference_as_i64;
use crate::shared::{
    CategoricalFeature1g1, CategoricalFeature2g1, DAlignment, Dna, FeatureError,
    InferenceParameters, InsertionFeature, VJAlignment,
};

use crate::vdj::Sequence;

use itertools::iproduct;

/// Contains the probability of the V gene ending at position e_v
/// For all reasonnable e_v
pub struct AggregatedFeatureEndV {
    pub index: usize,      // store the index of the V gene
    pub start_gene: usize, // store the start of the sequence in the V

    // deal with the range of possible values for endV
    pub start_v3: i64,
    pub end_v3: i64,

    // Contains all the likelihood
    likelihood: RangeArray1,

    // Dirty likelihood (will be updated as we go through the inference)
    dirty_likelihood: RangeArray1,
}

pub struct AggregatedFeatureStartJ {
    pub index: usize,     // store the index of the j gene
    pub start_seq: usize, // store the start of the J in the sequence

    // deal with the range of possible values for startJ
    pub start_j5: i64,
    pub end_j5: i64,

    // Contains all the likelihood
    likelihood: RangeArray1,

    // Dirty likelihood (will be updated as we go through the inference)
    dirty_likelihood: RangeArray1,
}

// Contains the probability of the D gene starting and ending position
pub struct AggregatedFeatureSpanD {
    pub index: usize, // store the index of the D gene

    // range of possible values
    pub start_d5: i64,
    pub end_d5: i64,
    pub start_d3: i64,
    pub end_d3: i64,

    // Contains all the likelihood  P(startD, endD | D)
    likelihood: RangeArray2,
    // Dirty likelihood, will be updated as we go through the inference
    dirty_likelihood: RangeArray2,
}

impl AggregatedFeatureEndV {
    pub fn new(
        v: &VJAlignment,
        feat_delv: &CategoricalFeature1g1,
        feat_error: &FeatureError,
        ip: &InferenceParameters,
    ) -> Option<AggregatedFeatureEndV> {
        let mut likelihood = RangeArray1::zeros((
            difference_as_i64(v.end_seq, feat_delv.dim().0) + 1,
            v.end_seq as i64 + 1,
        ));
        let mut total_likelihood = 0.;
        for delv in 0..feat_delv.dim().0 {
            let v_end = difference_as_i64(v.end_seq, delv);
            let ll_delv = feat_delv.likelihood((delv, v.index));
            let ll_v_err = feat_error.likelihood(v.errors(delv));
            let ll = ll_delv * ll_v_err;
            if ll > ip.min_likelihood {
                *likelihood.get_mut(v_end) = ll;
                total_likelihood += ll;
            }
        }

        if total_likelihood == 0. {
            return None;
        }

        Some(AggregatedFeatureEndV {
            start_v3: likelihood.min,
            end_v3: likelihood.max,
            dirty_likelihood: RangeArray1::zeros(likelihood.dim()),
            likelihood,
            index: v.index,
            start_gene: v.start_gene,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (i64, &f64)> + '_ {
        self.likelihood.iter().filter(|&(_, &v)| v != 0.0)
    }

    pub fn likelihood(&self, ev: i64) -> f64 {
        self.likelihood.get(ev)
    }

    pub fn max_likelihood(&self) -> f64 {
        self.likelihood.max_value()
    }

    pub fn dirty_update(&mut self, ev: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut(ev) += likelihood;
    }

    pub fn disaggregate(
        &self,
        v: &VJAlignment,
        feat_delv: &mut CategoricalFeature1g1,
        feat_error: &mut FeatureError,
        ip: &InferenceParameters,
    ) {
        for delv in 0..feat_delv.dim().0 {
            let v_end = difference_as_i64(v.end_seq, delv);
            let ll = feat_delv.likelihood((delv, v.index)) * feat_error.likelihood(v.errors(delv));

            if ll > ip.min_likelihood {
                let dirty_proba = self.dirty_likelihood.get(v_end); // P(ev)
                if dirty_proba > 0. {
                    // here we want to compute P(delV | V)
                    // at that point the V gene proba should be already updated
                    feat_delv.dirty_update((delv, v.index), dirty_proba);

                    feat_error.dirty_update(v.errors(delv), dirty_proba);
                }
            }
        }
    }
}

impl AggregatedFeatureStartJ {
    pub fn new(
        j: &VJAlignment,
        feat_delj: &CategoricalFeature1g1,
        feat_error: &FeatureError,
        ip: &InferenceParameters,
    ) -> Option<AggregatedFeatureStartJ> {
        let mut likelihood =
            RangeArray1::zeros((j.start_seq as i64, (j.start_seq + feat_delj.dim().0) as i64));

        let mut total_likelihood = 0.;
        for delj in 0..feat_delj.dim().0 {
            let j_start = (j.start_seq + delj) as i64;
            let ll_delj = feat_delj.likelihood((delj, j.index));
            let ll_errj = feat_error.likelihood(j.errors(delj));
            let ll = ll_delj * ll_errj;
            if ll > ip.min_likelihood {
                *likelihood.get_mut(j_start) = ll;
                total_likelihood += ll;
            }
        }

        if total_likelihood == 0. {
            return None;
        }
        Some(AggregatedFeatureStartJ {
            start_j5: likelihood.min,
            end_j5: likelihood.max,
            dirty_likelihood: RangeArray1::zeros(likelihood.dim()),
            likelihood,

            index: j.index,
            start_seq: j.start_seq,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (i64, &f64)> + '_ {
        self.likelihood.iter().filter(|&(_, &v)| v != 0.0)
    }

    pub fn likelihood(&self, sj: i64) -> f64 {
        self.likelihood.get(sj)
    }

    pub fn max_likelihood(&self) -> f64 {
        self.likelihood.max_value()
    }

    pub fn dirty_update(&mut self, sj: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut(sj) += likelihood;
    }

    pub fn disaggregate(
        &self,
        j: &VJAlignment,
        feat_delj: &mut CategoricalFeature1g1,
        feat_error: &mut FeatureError,
        ip: &InferenceParameters,
    ) {
        for delj in 0..feat_delj.dim().0 {
            let j_start = (j.start_seq + delj) as i64;
            let ll = feat_delj.likelihood((delj, j.index)) * feat_error.likelihood(j.errors(delj));
            if ll > ip.min_likelihood {
                let dirty_proba = self.dirty_likelihood.get(j_start);
                if dirty_proba > 0. {
                    feat_delj.dirty_update((delj, j.index), dirty_proba);

                    feat_error.dirty_update(j.errors(delj), dirty_proba)
                }
            }
        }
    }
}

impl AggregatedFeatureSpanD {
    pub fn new(
        ds: &Vec<DAlignment>,
        feat_deld: &CategoricalFeature2g1,
        feat_error: &FeatureError,
        ip: &InferenceParameters,
    ) -> Option<AggregatedFeatureSpanD> {
        if ds.is_empty() {
            return None;
        }

        let mut total_likelihood = 0.;
        let mut likelihoods = RangeArray2::zeros((
            (
                // min start, min end
                ds.iter().map(|x| x.pos).min().unwrap() as i64,
                ds.iter().map(|x| x.pos + x.len()).min().unwrap() as i64 - feat_deld.dim().1 as i64
                    + 1,
            ),
            (
                // max start, max end
                ds.iter().map(|x| x.pos).max().unwrap() as i64 + feat_deld.dim().0 as i64,
                ds.iter().map(|x| x.pos + x.len()).max().unwrap() as i64 + 1,
            ),
        ));

        let dindex = ds.first().unwrap().index;
        for d in ds {
            if d.index != dindex {
                panic!("AggregatedFeatureSpanD received different genes.");
            }

            for (deld5, deld3) in iproduct!(0..feat_deld.dim().0, 0..feat_deld.dim().1) {
                let d_start = (d.pos + deld5) as i64;
                let d_end = (d.pos + d.len() - deld3) as i64;
                if d_start > d_end {
                    continue;
                }

                let ll_deld = feat_deld.likelihood((deld5, deld3, d.index));
                let ll_errord = feat_error.likelihood(d.errors(deld5, deld3));
                let likelihood = ll_deld * ll_errord;

                if likelihood > ip.min_likelihood {
                    *likelihoods.get_mut((d_start, d_end)) += likelihood;
                    total_likelihood += likelihood;
                }
            }
        }

        if total_likelihood == 0. {
            return None;
        }

        Some(AggregatedFeatureSpanD {
            start_d5: likelihoods.min.0,
            end_d5: likelihoods.max.0,
            start_d3: likelihoods.min.1,
            end_d3: likelihoods.max.1,
            dirty_likelihood: RangeArray2::zeros(likelihoods.dim()),
            likelihood: likelihoods,
            index: dindex,
        })
    }

    pub fn likelihood(&self, sd: i64, ed: i64) -> f64 {
        self.likelihood.get((sd, ed))
    }

    pub fn iter(&self) -> impl Iterator<Item = (i64, i64, &f64)> + '_ {
        self.likelihood.iter() //.filter(|&(_, _, &v)| v != 0.0)
    }

    pub fn iter_fixed_dend(&self, dend: i64) -> impl Iterator<Item = (i64, &f64)> + '_ {
        let iteropt = if (dend < self.likelihood.min.1) || (dend >= self.likelihood.max.1) {
            None
        } else {
            Some(self.likelihood.iter_fixed_2nd(dend))
        };
        iteropt.into_iter().flatten()
        //  .filter(|&(_, &v)| v != 0.0)
    }

    pub fn max_likelihood(&self) -> f64 {
        self.likelihood.max_value()
    }

    pub fn dirty_update(&mut self, sd: i64, ed: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut((sd, ed)) += likelihood;
    }

    pub fn disaggregate(
        &self,
        ds: &[DAlignment],
        feat_deld: &mut CategoricalFeature2g1,
        feat_error: &mut FeatureError,
        ip: &InferenceParameters,
    ) {
        // Now with startD and end D
        for d in ds.iter() {
            for (deld5, deld3) in iproduct!(0..feat_deld.dim().0, 0..feat_deld.dim().1) {
                let d_start = (d.pos + deld5) as i64;
                let d_end = (d.pos + d.len() - deld3) as i64;
                if d_start > d_end {
                    continue;
                }
                let likelihood = feat_deld.likelihood((deld5, deld3, d.index))
                    * feat_error.likelihood(d.errors(deld5, deld3));

                if likelihood > ip.min_likelihood {
                    let dirty_proba = self.dirty_likelihood.get((d_start, d_end));
                    let corrected_proba =
                        dirty_proba * likelihood / self.likelihood(d_start, d_end);
                    if dirty_proba > 0. {
                        feat_deld.dirty_update((deld5, deld3, d.index), corrected_proba);

                        feat_error.dirty_update(d.errors(deld5, deld3), corrected_proba);
                    }
                }
            }
        }
    }
}

pub struct FeatureVD {
    likelihood: RangeArray2,
    dirty_likelihood: RangeArray2,
}

impl FeatureVD {
    pub fn new(
        sequence: &Sequence,
        feat_insvd: &InsertionFeature,
        delv_max: usize,
        deld5_max: usize,
        ip: &InferenceParameters,
    ) -> Option<FeatureVD> {
        if sequence.v_genes.is_empty() || sequence.d_genes.is_empty() {
            return None;
        }
        let min_end_v =
            sequence.v_genes.iter().map(|x| x.end_seq).min().unwrap() as i64 - delv_max as i64 + 1;
        let min_start_d = sequence.d_genes.iter().map(|x| x.pos).min().unwrap() as i64;
        let max_end_v = sequence.v_genes.iter().map(|x| x.end_seq).max().unwrap() as i64;
        let max_start_d =
            sequence.d_genes.iter().map(|x| x.pos).max().unwrap() as i64 + deld5_max as i64 - 1;

        let mut likelihoods =
            RangeArray2::zeros(((min_end_v, min_start_d), (max_end_v + 1, max_start_d + 1)));

        for ev in min_end_v..=max_end_v {
            for sd in min_start_d..=max_start_d {
                if sd >= ev && ((sd - ev) as usize) < feat_insvd.max_nb_insertions() {
                    let ins_vd_plus_first = sequence.get_subsequence(ev - 1, sd);
                    let likelihood = feat_insvd.likelihood(&ins_vd_plus_first);
                    if likelihood > ip.min_likelihood {
                        *likelihoods.get_mut((ev, sd)) = likelihood;
                    }
                }
            }
        }

        Some(FeatureVD {
            dirty_likelihood: RangeArray2::zeros(likelihoods.dim()),
            likelihood: likelihoods,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (i64, i64, &f64)> + '_ {
        self.likelihood.iter().filter(|&(_, _, &v)| v != 0.0)
    }

    pub fn max_ev(&self) -> i64 {
        self.likelihood.max.0
    }

    pub fn max_likelihood(&self) -> f64 {
        self.likelihood.max_value()
    }

    pub fn min_ev(&self) -> i64 {
        self.likelihood.min.0
    }

    pub fn max_sd(&self) -> i64 {
        self.likelihood.max.1
    }

    pub fn min_sd(&self) -> i64 {
        self.likelihood.min.1
    }

    pub fn likelihood(&self, ev: i64, sd: i64) -> f64 {
        self.likelihood.get((ev, sd))
    }

    pub fn dirty_update(&mut self, ev: i64, sd: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut((ev, sd)) += likelihood;
    }

    pub fn disaggregate(
        &self,
        sequence: &Dna,
        feat_insvd: &mut InsertionFeature,
        ip: &InferenceParameters,
    ) {
        for ev in self.likelihood.lower().0..self.likelihood.upper().0 {
            for sd in self.likelihood.lower().1..self.likelihood.upper().1 {
                if sd >= ev
                    && ((sd - ev) as usize) < feat_insvd.max_nb_insertions()
                    && self.likelihood(ev, sd) > ip.min_likelihood
                {
                    let ins_vd_plus_first = &sequence.extract_padded_subsequence(ev - 1, sd);
                    let likelihood = self.likelihood(ev, sd);
                    if likelihood > ip.min_likelihood {
                        feat_insvd
                            .dirty_update(ins_vd_plus_first, self.dirty_likelihood.get((ev, sd)))
                    }
                }
            }
        }
    }
}

pub struct FeatureDJ {
    likelihood: RangeArray2,
    dirty_likelihood: RangeArray2,
}

impl FeatureDJ {
    pub fn new(
        sequence: &Sequence,
        feat_insdj: &InsertionFeature,
        max_deld3: usize,
        max_delj: usize,
        ip: &InferenceParameters,
    ) -> Option<FeatureDJ> {
        if sequence.d_genes.is_empty() || sequence.j_genes.is_empty() {
            return None;
        }

        let min_end_d = sequence
            .d_genes
            .iter()
            .map(|x| x.pos + x.len())
            .min()
            .unwrap() as i64
            - max_deld3 as i64
            + 1;
        let min_start_j = sequence.j_genes.iter().map(|x| x.start_seq).min().unwrap() as i64;
        let max_end_d = sequence
            .d_genes
            .iter()
            .map(|x| x.pos + x.len())
            .max()
            .unwrap() as i64;
        let max_start_j = sequence.j_genes.iter().map(|x| x.start_seq).max().unwrap() as i64
            + max_delj as i64
            - 1;

        let mut likelihoods =
            RangeArray2::zeros(((min_end_d, min_start_j), (max_end_d + 1, max_start_j + 1)));

        for ed in min_end_d..=max_end_d {
            for sj in min_start_j..=max_start_j {
                if sj >= ed && ((sj - ed) as usize) < feat_insdj.max_nb_insertions() {
                    // careful we need to reverse ins_dj for the inference
                    let mut ins_dj_plus_last = sequence.get_subsequence(ed, sj + 1);
                    ins_dj_plus_last.reverse();
                    let likelihood = feat_insdj.likelihood(&ins_dj_plus_last);
                    if likelihood > ip.min_likelihood {
                        *likelihoods.get_mut((ed, sj)) = likelihood;
                    }
                }
            }
        }

        Some(FeatureDJ {
            dirty_likelihood: RangeArray2::zeros(likelihoods.dim()),
            likelihood: likelihoods,
        })
    }

    pub fn likelihood(&self, ed: i64, sj: i64) -> f64 {
        self.likelihood.get((ed, sj))
    }

    pub fn max_likelihood(&self) -> f64 {
        self.likelihood.max_value()
    }

    pub fn iter(&self) -> impl Iterator<Item = (i64, i64, &f64)> + '_ {
        self.likelihood.iter().filter(|&(_, _, &v)| v != 0.0)
    }

    pub fn max_ed(&self) -> i64 {
        self.likelihood.max.0
    }

    pub fn min_ed(&self) -> i64 {
        self.likelihood.min.0
    }

    pub fn max_sj(&self) -> i64 {
        self.likelihood.max.1
    }

    pub fn min_sj(&self) -> i64 {
        self.likelihood.min.1
    }

    pub fn dirty_update(&mut self, ed: i64, sj: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut((ed, sj)) += likelihood;
    }

    pub fn disaggregate(
        &self,
        sequence: &Dna,
        feat_insdj: &mut InsertionFeature,
        ip: &InferenceParameters,
    ) {
        for ed in self.likelihood.lower().0..self.likelihood.upper().0 {
            for sj in self.likelihood.lower().1..self.likelihood.upper().1 {
                if sj >= ed
                    && ((sj - ed) as usize) < feat_insdj.max_nb_insertions()
                    && (self.dirty_likelihood.get((ed, sj)) > 0.)
                {
                    let mut ins_dj_plus_last = sequence.extract_padded_subsequence(ed, sj + 1);
                    ins_dj_plus_last.reverse();
                    let likelihood = feat_insdj.likelihood(&ins_dj_plus_last);
                    if likelihood > ip.min_likelihood {
                        feat_insdj
                            .dirty_update(&ins_dj_plus_last, self.dirty_likelihood.get((ed, sj)));
                    }
                }
            }
        }
    }
}
