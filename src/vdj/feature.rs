use crate::shared::feature::Feature;
use crate::shared::sequence::SequenceType;

use crate::shared::utils::difference_as_i64;
use crate::shared::{
    data_structures::RangeArray1, data_structures::RangeArray2, CategoricalFeature1g1,
    CategoricalFeature2g1, DAlignment, DnaLike, ErrorDAlignment, ErrorJAlignment, ErrorVAlignment,
    FeatureError, InfEvent, InferenceParameters, InsertionFeature, Likelihood,
    Likelihood1DContainer, Likelihood2DContainer, LikelihoodInsContainer, VJAlignment,
};
use crate::vdj::Sequence;
use itertools::iproduct;
use std::sync::Arc;

/// Contains the probability of the V gene ending at position e_v
/// For all reasonnable e_v
#[derive(Debug)]
pub struct AggregatedFeatureEndV {
    pub index: usize,      // store the index of the V gene
    pub start_gene: usize, // store the start of the sequence in the V

    // deal with the range of possible values for endV
    pub start_v3: i64,
    pub end_v3: i64,

    // alignment
    pub alignment: Arc<VJAlignment>,

    // Contains all the likelihood
    likelihood: Likelihood1DContainer,
    // Dirty likelihood (will be updated as we go through the inference)
    dirty_likelihood: RangeArray1,
}

#[derive(Debug)]
pub struct AggregatedFeatureStartJ {
    pub index: usize,   // store the index of the j gene
    pub start_seq: i64, // store the start of the J in the sequence

    // deal with the range of possible values for startJ
    pub start_j5: i64,
    pub end_j5: i64,
    //    start_j_in_gene: i64,

    // alignment
    pub alignment: Arc<VJAlignment>,

    // Contains all the likelihood
    likelihood: Likelihood1DContainer,

    // Dirty likelihood (will be updated as we go through the inference)
    dirty_likelihood: RangeArray1,
}

/// Contains the probability of the D gene starting and ending position
#[derive(Debug)]
pub struct AggregatedFeatureSpanD {
    pub index: usize, // store the index of the D gene

    // range of possible values
    pub start_d5: i64,
    pub end_d5: i64,
    pub start_d3: i64,
    pub end_d3: i64,

    // Contains all the likelihood  P(startD, endD | D)
    likelihood: Likelihood2DContainer,
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
        let mut likelihoods = Likelihood1DContainer::zeros(
            difference_as_i64(v.end_seq, feat_delv.dim().0) + 1,
            v.end_seq as i64 + 1,
            ip.likelihood_type,
        );

        let mut total_likelihood = 0.;
        for delv in 0..feat_delv.dim().0 {
            let v_end = difference_as_i64(v.end_seq, delv);
            let ll_delv = feat_delv.likelihood((delv, v.index));
            let ll_v_err = feat_error.likelihood(v.errors(delv, 0));
            let ll = ll_delv * ll_v_err;
            if ll > ip.min_likelihood {
                let likelihood = match ip.likelihood_type {
                    SequenceType::Dna => Likelihood::Scalar(ll),
                    SequenceType::Protein => Likelihood::from_v_side(v, delv) * ll,
                };
                likelihoods.add_to(v_end, likelihood);
                total_likelihood += ll;
            }
        }

        if total_likelihood == 0. {
            return None;
        }

        Some(AggregatedFeatureEndV {
            //            len_v_in_gene: v.end_seq as i64,
            start_v3: difference_as_i64(v.end_seq, feat_delv.dim().0) + 1,
            end_v3: v.end_seq as i64 + 1,
            dirty_likelihood: RangeArray1::zeros(likelihoods.dim()),
            likelihood: likelihoods,
            index: v.index,
            start_gene: v.start_gene,
            alignment: Arc::new(v.clone()),
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (i64, Likelihood)> + '_ {
        self.likelihood.iter().filter(|(_, v)| !v.is_zero())
    }

    pub fn likelihood(&self, ev: i64) -> Likelihood {
        self.likelihood.get(ev)
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
        debug_assert!(ip.infer_features);

        for delv in 0..feat_delv.dim().0 {
            let v_end = difference_as_i64(v.end_seq, delv);
            let ll =
                feat_delv.likelihood((delv, v.index)) * feat_error.likelihood(v.errors(delv, 0));

            if ll > ip.min_likelihood {
                let dirty_proba = self.dirty_likelihood.get(v_end); // P(ev)
                if dirty_proba > 0. {
                    // here we want to compute P(delV | V)
                    // at that point the V gene proba should be already updated
                    feat_delv.dirty_update((delv, v.index), dirty_proba);
                    feat_error.dirty_update_v_fragment(
                        ErrorVAlignment { val: &v, del: delv },
                        dirty_proba,
                    );
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
        let mut likelihoods = Likelihood1DContainer::zeros(
            j.start_seq as i64 - j.start_gene as i64,
            j.start_seq as i64 - j.start_gene as i64 + feat_delj.dim().0 as i64,
            ip.likelihood_type,
        );

        let mut total_likelihood = 0.;

        for delj in 0..feat_delj.dim().0 {
            let j_start = j.start_seq as i64 - j.start_gene as i64 + delj as i64;
            let ll_delj = feat_delj.likelihood((delj, j.index));
            let ll_errj = feat_error.likelihood(j.errors(0, delj));
            let ll = ll_delj * ll_errj;
            if ll > ip.min_likelihood {
                let likelihood = match ip.likelihood_type {
                    SequenceType::Dna => Likelihood::Scalar(ll),
                    SequenceType::Protein => Likelihood::from_j_side(j, delj) * ll,
                };
                likelihoods.add_to(j_start, likelihood);
                total_likelihood += ll;
            }
        }

        if total_likelihood == 0. {
            return None;
        }
        Some(AggregatedFeatureStartJ {
            //            start_j_in_gene: j.start_seq as i64 - j.start_gene as i64,
            start_j5: j.start_seq as i64 - j.start_gene as i64,
            end_j5: j.start_seq as i64 - j.start_gene as i64 + feat_delj.dim().0 as i64,
            dirty_likelihood: RangeArray1::zeros(likelihoods.dim()),
            likelihood: likelihoods,
            index: j.index,
            start_seq: j.start_seq as i64 - j.start_gene as i64,
            alignment: Arc::new(j.clone()),
            //first_nucleotides,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (i64, Likelihood)> + '_ {
        self.likelihood.iter().filter(|(_, v)| !v.is_zero())
    }

    pub fn likelihood(&self, sj: i64) -> Likelihood {
        self.likelihood.get(sj)
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
            let j_start = j.start_seq as i64 - j.start_gene as i64 + delj as i64;
            let ll =
                feat_delj.likelihood((delj, j.index)) * feat_error.likelihood(j.errors(0, delj));
            if ll > ip.min_likelihood {
                let dirty_proba = self.dirty_likelihood.get(j_start);
                if dirty_proba > 0. {
                    if ip.infer_features {
                        feat_delj.dirty_update((delj, j.index), dirty_proba);

                        feat_error.dirty_update_j_fragment(
                            ErrorJAlignment { jal: &j, del: delj },
                            dirty_proba,
                        )
                    }
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

        let min_start = ds.iter().map(|x| x.pos).min().unwrap() as i64;
        let min_end =
            ds.iter().map(|x| x.pos + x.len() as i64).min().unwrap() - feat_deld.dim().1 as i64 + 1;
        let max_start = ds.iter().map(|x| x.pos).max().unwrap() as i64 + feat_deld.dim().0 as i64;
        let max_end = ds.iter().map(|x| x.pos + x.len() as i64).max().unwrap() + 1;

        let mut likelihoods = Likelihood2DContainer::zeros(
            (min_start, min_end),
            (max_start, max_end),
            ip.likelihood_type,
        );

        let dindex = ds.first().unwrap().index;
        for d in ds {
            debug_assert!(d.index == dindex);

            for (deld5, deld3) in iproduct!(0..feat_deld.dim().0, 0..feat_deld.dim().1) {
                let d_start = d.pos + deld5 as i64;
                let d_end = d.pos + (d.len() - deld3) as i64;
                if d_start > d_end {
                    continue;
                }

                let ll_deld = feat_deld.likelihood((deld5, deld3, d.index));
                let ll_errord = feat_error.likelihood(d.errors(deld5, deld3));

                let ll = ll_deld * ll_errord;
                if ll > ip.min_likelihood {
                    let likelihood = match ip.likelihood_type {
                        SequenceType::Protein => Likelihood::from_d_sides(d, deld5, deld3) * ll,
                        SequenceType::Dna => Likelihood::Scalar(ll),
                    };

                    likelihoods.add_to((d_start, d_end), likelihood);
                    total_likelihood += ll;
                }
            }
        }

        if total_likelihood == 0. {
            return None;
        }

        Some(AggregatedFeatureSpanD {
            start_d5: likelihoods.min().0,
            end_d5: likelihoods.max().0,
            start_d3: likelihoods.min().1,
            end_d3: likelihoods.max().1,
            dirty_likelihood: RangeArray2::zeros(likelihoods.dim()),
            likelihood: likelihoods,
            index: dindex,
        })
    }

    pub fn likelihood(&self, sd: i64, ed: i64) -> Likelihood {
        self.likelihood.get((sd, ed))
    }

    pub fn iter(&self) -> impl Iterator<Item = (i64, i64, Likelihood)> + '_ {
        self.likelihood.iter() //.filter(|&(_, _, &v)| v != 0.0)
    }

    pub fn iter_fixed_dend(&self, dend: i64) -> impl Iterator<Item = (i64, Likelihood)> + '_ {
        let iteropt = if (dend < self.likelihood.min().1) || (dend >= self.likelihood.max().1) {
            None
        } else {
            Some(self.likelihood.iter_fixed_2nd(dend))
        };
        iteropt.into_iter().flatten()
        //  .filter(|&(_, &v)| v != 0.0)
    }

    // pub fn max_likelihood(&self) -> f64 {
    //     self.likelihood.max_value()
    // }

    pub fn dirty_update(&mut self, sd: i64, ed: i64, likelihood: f64) {
        *self.dirty_likelihood.get_mut((sd, ed)) += likelihood;
    }

    pub fn disaggregate(
        &self,
        ds: &[DAlignment],
        feat_deld: &mut CategoricalFeature2g1,
        feat_error: &mut FeatureError,
        event: &mut Option<InfEvent>,
        ip: &InferenceParameters,
    ) {
        // disaggregate should only work with the scalar version

        let mut best_likelihood = 0.;
        // Now with startD and end D
        for d in ds.iter() {
            for (deld5, deld3) in iproduct!(0..feat_deld.dim().0, 0..feat_deld.dim().1) {
                let d_start = d.pos + deld5 as i64;
                let d_end = d.pos + (d.len() - deld3) as i64;
                if d_start > d_end {
                    continue;
                }
                let likelihood = feat_deld.likelihood((deld5, deld3, d.index))
                    * feat_error.likelihood(d.errors(deld5, deld3));

                if likelihood > ip.min_likelihood {
                    let dirty_proba = self.dirty_likelihood.get((d_start, d_end));
                    let corrected_proba = dirty_proba * likelihood
                        / self.likelihood(d_start, d_end).to_scalar().unwrap();
                    if dirty_proba <= 0. && ip.infer_features {
                        continue;
                    }
                    if ip.infer_features {
                        feat_deld.dirty_update((deld5, deld3, d.index), corrected_proba);

                        feat_error.dirty_update_d_fragment(
                            ErrorDAlignment {
                                dal: &d,
                                deld5,
                                deld3,
                            },
                            corrected_proba,
                        );
                    }

                    if ip.store_best_event {
                        //if likelihood / self.likelihood(d_start, d_end) > best_proba {

                        if let Some(ev) = event {
                            if likelihood > best_likelihood {
                                // If this is the most likely d
                                if (ev.d_index == d.index)
                                    && (ev.start_d == d_start)
                                    && (ev.end_d == d_end)
                                {
                                    ev.pos_d = d.pos as i64;
                                    //best_proba = likelihood / self.likelihood(d_start, d_end);
                                    best_likelihood = likelihood;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct FeatureVD {
    // (end_v, start_d, first_nuc)
    likelihood: LikelihoodInsContainer,
    dirty_likelihood: LikelihoodInsContainer,
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

        let mut likelihoods = LikelihoodInsContainer::zeros(
            (min_end_v, min_start_d),
            (max_end_v + 1, max_start_d + 1),
            ip.likelihood_type,
        );

        for ev in min_end_v..=max_end_v {
            for sd in min_start_d..=max_start_d {
                // sd must be larger than 0 (the sequence should contains at least a bit of V)
                if sd >= 0
                    && sd < sequence.sequence.len() as i64
                    && sd >= ev
                    && ((sd - ev) as usize) < feat_insvd.max_nb_insertions()
                {
                    let ins_vd = sequence.get_subsequence(ev, sd);
                    for first_nucleotide in 0..4 {
                        let likelihood = feat_insvd.likelihood(&ins_vd, first_nucleotide);
                        // let likelihood = match ip.likelihood_type {
                        //     SequenceType::Protein => Likelihood::from_insertions(&ins_vd) * ll,
                        //     SequenceType::Dna => Likelihood::Scalar(ll),
                        // };
                        // if ev == 4 && sd == 5 {
                        //     println!("AAH {:?}", likelihood.max());
                        // }

                        if likelihood.max() > ip.min_likelihood {
                            likelihoods.add_to((ev, sd), first_nucleotide, likelihood);
                        }
                    }
                }
            }
        }

        Some(FeatureVD {
            dirty_likelihood: LikelihoodInsContainer::zeros(
                likelihoods.dim().0,
                likelihoods.dim().1,
                ip.likelihood_type,
            ),
            likelihood: likelihoods,
        })
    }

    // /// sequence is the inserted sequence
    // fn precompute_likelihood(
    //     insfeat: &InsertionFeature,
    //     sequence: &DnaLike,
    //     first_nucleotide: usize,
    //     ip: &InferenceParameters,
    // ) -> Likelihood {
    //     return match ip.likelihood_type {
    //         Matrix => {
    //             let matrix = Matrix16::zeros();
    //             for (left, right, seq) in sequence.fix_extremities() {
    //                 matrix[[left, right]] = insfeat.likelihood(seq, first_nucleotide);
    //             }
    //             Likelihood::Matrix(matrix)
    //         }
    //         Float => Likelihood::Scalar(insfeat.likelihood(sequence, first_nucleotide)),
    //     };
    // }

    pub fn max_ev(&self) -> i64 {
        self.likelihood.max().0
    }

    pub fn min_ev(&self) -> i64 {
        self.likelihood.min().0
    }

    pub fn max_sd(&self) -> i64 {
        self.likelihood.max().1
    }

    pub fn min_sd(&self) -> i64 {
        self.likelihood.min().1
    }

    pub fn likelihood(&self, ev: i64, sd: i64, previous_nuc: usize) -> Likelihood {
        self.likelihood.get((ev, sd), previous_nuc)
    }

    pub fn dirty_update(&mut self, ev: i64, sd: i64, previous_nuc: usize, likelihood: f64) {
        self.dirty_likelihood
            .add_to((ev, sd), previous_nuc, Likelihood::Scalar(likelihood));
    }

    pub fn disaggregate(
        &self,
        sequence: &DnaLike,
        feat_insvd: &mut InsertionFeature,
        ip: &InferenceParameters,
    ) {
        if !ip.infer_features {
            return;
        }
        // disaggregate only works with scalar
        for ev in self.likelihood.min().0..self.likelihood.max().0 {
            for sd in self.likelihood.min().1..self.likelihood.max().1 {
                if sd >= 0
                    && sd < sequence.len() as i64
                    && sd >= ev
                    && ((sd - ev) as usize) < feat_insvd.max_nb_insertions()
                {
                    //let ins_vd_plus_first = &sequence.extract_padded_subsequence(ev - 1, sd);
                    // let ins_vd_plus_first = &sequence.extract_padded_subsequence(ev - 1, sd);
                    let ins_vd = &sequence.extract_padded_subsequence(ev, sd);
                    for previous_nucleotide in 0..4 {
                        let ll = self
                            .likelihood(ev, sd, previous_nucleotide)
                            .to_scalar()
                            .unwrap();
                        let updated_ll = self
                            .dirty_likelihood
                            .get((ev, sd), previous_nucleotide)
                            .to_scalar()
                            .unwrap();
                        if ll > ip.min_likelihood && updated_ll > 0. {
                            feat_insvd.dirty_update(ins_vd, previous_nucleotide, updated_ll)
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct FeatureDJ {
    likelihood: LikelihoodInsContainer,
    dirty_likelihood: LikelihoodInsContainer,
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

        let ((min_end_d, max_end_d), (min_start_j, max_start_j)) =
            Self::span_end_start(sequence, max_deld3, max_delj);
        let mut likelihoods = LikelihoodInsContainer::zeros(
            (min_end_d, min_start_j),
            (max_end_d + 1, max_start_j + 1),
            ip.likelihood_type,
        );

        for ed in min_end_d..=max_end_d {
            for sj in min_start_j..=max_start_j {
                if ed >= 0
                    && ed < sequence.sequence.len() as i64
                    && sj >= ed
                    && ((sj - ed) as usize) < feat_insdj.max_nb_insertions()
                {
                    let ins_dj = sequence.get_subsequence(ed, sj);
                    //let mut ins_dj_rev = ins_dj.clone();
                    //                    ins_dj_rev.reverse();

                    if ed == 0 && sj == 2 {
                        println!("{:?} {:?}", ins_dj, ins_dj);
                    }

                    for first_nucleotide in 0..4 {
                        let likelihood = feat_insdj.likelihood(&ins_dj, first_nucleotide);
                        // let likelihood = match ip.likelihood_type {
                        //     SequenceType::Protein => Likelihood::from_insertions(&ins_dj) * ll,
                        //     SequenceType::Dna => Likelihood::Scalar(ll),
                        // };
                        // //                        if ins_dj.len() > 0 {
                        // // println!("{} {} {:?} {:?}", ed, sj, ins_dj.to_dna(), likelihood);
                        // //                        }
                        if likelihood.max() > ip.min_likelihood {
                            likelihoods.add_to((ed, sj), first_nucleotide, likelihood);
                        }
                    }
                }
            }
        }

        Some(FeatureDJ {
            dirty_likelihood: LikelihoodInsContainer::zeros(
                likelihoods.dim().0,
                likelihoods.dim().1,
                ip.likelihood_type,
            ),
            likelihood: likelihoods,
        })
    }

    /// sequence is the insertion sequence reversed
    // fn precompute_likelihood(
    //     insfeat: &InsertionFeature,
    //     sequence: &DnaLike,
    //     first_nucleotide: usize,
    //     ip: &InferenceParameters,
    // ) -> Likelihood {
    //     // careful we need to reverse ins_dj for the inference
    //     return match ip.likelihood_type {
    //         LikelihoodType::Matrix | LikelihoodType::Vector => {
    //             unimplemented!("Not yet!");
    //             // let matrix = Matrix16::zeros();
    //             // for (left, right, seq) in sequence.fix_extremities() {
    //             //     matrix[[left, right]] = insfeat.likelihood(seq, first_nucleotide);
    //             // }
    //             // Likelihood::Matrix(matrix)
    //         }
    //         LikelihoodType::Scalar => {
    //             Likelihood::Scalar(insfeat.likelihood(sequence, first_nucleotide))
    //         }
    //     };
    // }

    pub fn span_end_start(
        sequence: &Sequence,
        max_deld3: usize,
        max_delj: usize,
    ) -> ((i64, i64), (i64, i64)) {
        let min_end_d = sequence
            .d_genes
            .iter()
            .map(|x| x.pos + x.len() as i64)
            .min()
            .unwrap()
            - max_deld3 as i64
            + 1;
        let min_start_j = sequence
            .j_genes
            .iter()
            .map(|x| x.start_seq as i64 - x.start_gene as i64)
            .min()
            .unwrap() as i64;
        let max_end_d = sequence
            .d_genes
            .iter()
            .map(|x| x.pos + x.len() as i64)
            .max()
            .unwrap();
        let max_start_j = sequence
            .j_genes
            .iter()
            .map(|x| x.start_seq as i64 - x.start_gene as i64)
            .max()
            .unwrap() as i64
            + max_delj as i64
            - 1;
        ((min_end_d, max_end_d), (min_start_j, max_start_j))
    }

    pub fn likelihood(&self, ed: i64, sj: i64, first_nucleotide: usize) -> Likelihood {
        self.likelihood.get((ed, sj), first_nucleotide)
    }

    // pub fn max_likelihood(&self) -> f64 {
    //     self.likelihood.max_value()
    // }

    pub fn iter(&self) -> impl Iterator<Item = (usize, i64, i64, Likelihood)> + '_ {
        self.likelihood.iter().filter(|(_, _, _, v)| !v.is_zero())
    }

    pub fn max_ed(&self) -> i64 {
        self.likelihood.max().0
    }

    pub fn min_ed(&self) -> i64 {
        self.likelihood.min().0
    }

    pub fn max_sj(&self) -> i64 {
        self.likelihood.max().1
    }

    pub fn min_sj(&self) -> i64 {
        self.likelihood.min().1
    }

    pub fn dirty_update(&mut self, ed: i64, sj: i64, previous_nuc: usize, likelihood: f64) {
        self.dirty_likelihood
            .add_to((ed, sj), previous_nuc, Likelihood::Scalar(likelihood));
    }

    pub fn disaggregate(
        &self,
        sequence: &DnaLike,
        feat_insdj: &mut InsertionFeature,
        ip: &InferenceParameters,
    ) {
        if !ip.infer_features {
            return;
        }
        for ed in self.likelihood.min().0..self.likelihood.max().0 {
            for sj in self.likelihood.min().1..self.likelihood.max().1 {
                if ed >= 0
                    && ed < sequence.len() as i64
                    && sj >= ed
                    && ((sj - ed) as usize) < feat_insdj.max_nb_insertions()
                {
                    let ins_dj = sequence.extract_padded_subsequence(ed, sj);
                    //ins_dj.reverse();
                    for next_nucleotide in 0..4 {
                        let ll = self
                            .likelihood(ed, sj, next_nucleotide)
                            .to_scalar()
                            .unwrap();
                        let updated_ll = self
                            .dirty_likelihood
                            .get((ed, sj), next_nucleotide)
                            .to_scalar()
                            .unwrap();
                        if ll > ip.min_likelihood && updated_ll > 0. {
                            feat_insdj.dirty_update(&ins_dj, next_nucleotide, updated_ll);
                        }
                    }
                }
            }
        }
    }
}
