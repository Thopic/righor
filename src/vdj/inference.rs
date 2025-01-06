use crate::shared::utils::difference_as_i64;
use crate::shared::Modelable;
use crate::shared::{errors::FeatureError, ErrorParameters, InferenceParameters};
use crate::shared::{
    feature::{
        CategoricalFeature1g1, CategoricalFeature2g1, CategoricalFeature3, Feature, InfEvent,
        InsertionFeature, ResultInference,
    },
    DNAMarkovChain,
};
use crate::shared::{ErrorDAlignment, ErrorJAlignment, ErrorVAlignment};
use crate::vdj::{
    AggregatedFeatureEndV, AggregatedFeatureSpanD, AggregatedFeatureStartJ, FeatureDJ, FeatureVD,
    Model, Sequence,
};
use anyhow::Result;

use std::cmp;
use std::sync::Arc;

#[derive(Default, Clone, Debug)]
pub struct Features {
    pub delv: CategoricalFeature1g1,
    pub vdj: CategoricalFeature3,
    pub delj: CategoricalFeature1g1,
    pub deld: CategoricalFeature2g1, // d5, d3, d
    pub insvd: InsertionFeature,
    pub insdj: InsertionFeature,
    pub error: FeatureError,
    pub log_likelihood: Option<f64>,
}

impl Features {
    pub fn new(model: &Model) -> Result<Features> {
        Ok(Features {
            vdj: CategoricalFeature3::new(&model.p_vdj)?,
            delv: CategoricalFeature1g1::new(&model.p_del_v_given_v)?,
            delj: CategoricalFeature1g1::new(&model.p_del_j_given_j)?,
            deld: CategoricalFeature2g1::new(&model.p_del_d5_del_d3)?, // dim: (d5, d3, d)
            insvd: InsertionFeature::new(&model.p_ins_vd, Arc::clone(&model.markov_chain_vd))?,
            insdj: InsertionFeature::new(&model.p_ins_dj, Arc::clone(&model.markov_chain_dj))?,
            error: model.error.get_feature()?,
            log_likelihood: None,
        })
    }

    /// Update the model from a vector of features and "average" the features.
    /// Return the new features and the total log likelihood
    pub fn update(
        features: Vec<Features>,
        model: &mut Model,
        ip: &InferenceParameters,
    ) -> Result<(Vec<Features>, f64)> {
        let errors = &mut ErrorParameters::update_error(
            features.iter().map(|a| a.error.clone()).collect(),
            &mut model.error,
        )?;

        let insvd = InsertionFeature::average(
            features
                .iter()
                .zip(errors.iter())
                .map(|(f, e)| f.insvd.correct_for_error(e).clone()),
        )?;
        let insdj = InsertionFeature::average(
            features
                .iter()
                .zip(errors.iter())
                .map(|(f, e)| f.insdj.correct_for_error(e).clone()),
        )?;

        let delv = CategoricalFeature1g1::average(features.iter().map(|a| a.delv.clone()))?;
        let delj = CategoricalFeature1g1::average(features.iter().map(|a| a.delj.clone()))?;
        let deld = CategoricalFeature2g1::average(features.iter().map(|a| a.deld.clone()))?;
        let vdj = CategoricalFeature3::average(features.iter().map(|a| a.vdj.clone()))?;

        if ip.infer_features.genes {
            model.set_p_vdj(&vdj.clone().probas)?;
        }
        if ip.infer_features.del_v {
            model.p_del_v_given_v = delv.clone().probas;
        }
        if ip.infer_features.del_j {
            model.p_del_j_given_j = delj.clone().probas;
        }
        if ip.infer_features.del_d {
            model.p_del_d5_del_d3 = deld.clone().probas;
        }

        let (p_vd, mc_vd) = insvd.get_parameters();
        if ip.infer_features.ins_vd {
            model.p_ins_vd = p_vd;
            model.markov_chain_vd = Arc::new(DNAMarkovChain::new(&mc_vd, false)?);
        }
        let (p_dj, mc_dj) = insdj.get_parameters();
        if ip.infer_features.ins_dj {
            model.p_ins_dj = p_dj;
            model.markov_chain_dj = Arc::new(DNAMarkovChain::new(&mc_dj, true)?);
        }

        let sum_log_likelihood = features.iter().map(|x| x.log_likelihood.unwrap()).sum();

        // Now update the features vector
        let mut new_features = Vec::new();
        for error in errors {
            new_features.push(Features {
                vdj: vdj.clone(),
                delv: delv.clone(),
                delj: delj.clone(),
                deld: deld.clone(),
                insvd: insvd.clone(),
                insdj: insdj.clone(),
                error: error.clone(),
                log_likelihood: None,
            });
        }
        model.initialize()?;
        Ok((new_features, sum_log_likelihood))
    }

    /// Core function, iterate over all realistic scenarios to compute the
    /// likelihood of the sequence and update the parameters
    pub fn infer(
        &mut self,
        sequence: &Sequence,
        ip: &InferenceParameters,
    ) -> Result<ResultInference> {
        // println!("VAL {:?}", sequence.v_genes[0]);
        // println!("JAL {:?}", sequence.j_genes[0]);

        // small positive likelihood, in case the inference stops early
        self.log_likelihood = Some((ip.min_likelihood).log2());

        // Estimate the likelihood of all possible insertions
        let Some(mut ins_vd) = FeatureVD::new(
            sequence,
            &self.insvd,
            self.delv.dim().0,
            self.deld.dim().0,
            ip,
        ) else {
            return Ok(ResultInference::impossible());
        };
        let Some(mut ins_dj) = FeatureDJ::new(
            sequence,
            &self.insdj,
            self.deld.dim().1,
            self.delj.dim().0,
            ip,
        ) else {
            return Ok(ResultInference::impossible());
        };

        // Define the aggregated features for this sequence:
        let mut features_d = Vec::new();
        for d_idx in 0..self.vdj.dim().1 {
            let feature_d = AggregatedFeatureSpanD::new(
                &sequence.get_specific_dgene(d_idx),
                &self.deld,
                &self.error,
                ip,
            );
            features_d.push(feature_d);
        }

        let mut features_v = Vec::new();
        for val in &sequence.v_genes {
            let feature_v = AggregatedFeatureEndV::new(val, &self.delv, &self.error, ip);
            features_v.push(feature_v);
        }

        let mut features_j = Vec::new();
        for jal in &sequence.j_genes {
            let feature_j = AggregatedFeatureStartJ::new(jal, &self.delj, &self.error, ip);
            features_j.push(feature_j);
        }

        let mut result = ResultInference::impossible();

        // Main loop
        for v in features_v.iter_mut().filter_map(|x| x.as_mut()) {
            for j in features_j.iter_mut().filter_map(|x| x.as_mut()) {
                for d in features_d.iter_mut().filter_map(|x| x.as_mut()) {
                    self.infer_given_vdj(v, d, j, &mut ins_vd, &mut ins_dj, ip, &mut result)?;
                }
            }
        }

        // add a small positive likelihood to deal with the case where result.likelihood is 0.
        self.log_likelihood = Some((result.likelihood + ip.min_likelihood).log2());

        if !ip.need_disaggregate() {
            return Ok(result);
        }

        // disaggregate the insertion features
        ins_vd.disaggregate(&sequence.sequence, &mut self.insvd, ip);
        ins_dj.disaggregate(&sequence.sequence, &mut self.insdj, ip);

        // disaggregate the v/d/j features
        for (val, v) in sequence.v_genes.iter().zip(features_v.iter_mut()) {
            match v {
                Some(f) => f.disaggregate(val, &mut self.delv, &mut self.error, ip),
                None => continue,
            }
        }
        for (jal, j) in sequence.j_genes.iter().zip(features_j.iter_mut()) {
            match j {
                Some(f) => f.disaggregate(jal, &mut self.delj, &mut self.error, ip),
                None => continue,
            }
        }

        for d in &mut features_d {
            match d {
                Some(f) => f.disaggregate(
                    &sequence.get_specific_dgene(f.index),
                    &mut self.deld,
                    &mut self.error,
                    &mut result.best_event,
                    ip,
                ),
                None => continue,
            }
        }

        // Divide all the proba by P(R) (the probability of the sequence)
        if result.likelihood > 0. {
            self.scale(result.likelihood)?;
        } else {
            return Ok(ResultInference::impossible());
        }

        // Return the result
        Ok(result)
    }
}

impl Features {
    /// Brute-force inference
    /// for test-purpose only
    pub fn infer_brute_force(
        &mut self,
        sequence: &Sequence,
        ip: &InferenceParameters,
    ) -> Result<ResultInference> {
        debug_assert!(!sequence.sequence.is_protein());
        let mut result = ResultInference::impossible();

        // Main loop
        for val in sequence.v_genes.clone() {
            for jal in sequence.j_genes.clone() {
                for dal in sequence.d_genes.clone() {
                    for delv in 0..self.delv.dim().0 {
                        for delj in 0..self.delj.dim().0 {
                            for deld5 in 0..self.deld.dim().0 {
                                for deld3 in 0..self.deld.dim().1 {
                                    let d_start = dal.pos + deld5 as i64;
                                    let d_end = dal.pos + (dal.len() - deld3) as i64;
                                    let j_start =
                                        jal.start_seq as i64 - jal.start_gene as i64 + delj as i64;
                                    let v_end = difference_as_i64(val.end_seq, delv);
                                    if (d_start > d_end) || (j_start < d_end) || (d_start < v_end) {
                                        continue;
                                    }
                                    if (d_start < 0)
                                        || (d_start >= sequence.sequence.len() as i64)
                                        || (d_end < 0)
                                        || (d_end > sequence.sequence.len() as i64)
                                    {
                                        continue;
                                    }

                                    let ins_dj = sequence.get_subsequence(d_end, j_start);
                                    let ins_vd = sequence.get_subsequence(v_end, d_start);

                                    let last_v_nucleotide = val.get_last_nucleotide(delv);
                                    let first_j_nucleotide = jal.get_first_nucleotide(delj);

                                    // let nb_errors = val.nb_errors(delv)
                                    //     + jal.nb_errors(delj)
                                    //     + dal.nb_errors(deld5, deld3);

                                    let ll = self.vdj.likelihood((val.index, dal.index, jal.index))
                                        * self.delv.likelihood((delv, val.index))
                                        * self.delj.likelihood((delj, jal.index))
                                        * self.deld.likelihood((deld5, deld3, dal.index))
                                        * self
                                            .insdj
                                            .likelihood(&ins_dj, first_j_nucleotide)
                                            .to_scalar()
                                        * self
                                            .insvd
                                            .likelihood(&ins_vd, last_v_nucleotide)
                                            .to_scalar()
                                        * self.error.likelihood(val.errors(delv, 0))
                                        * self.error.likelihood(jal.errors(0, delj))
                                        * self.error.likelihood(dal.errors(deld5, deld3));

                                    // println!(
                                    //     "{:.1e}\t{:.1e}\t{:.1e}\t{:.1e}\t{:.1e}\t{:.1e}\t{:.1e}",
                                    //     self.vdj.likelihood((val.index, dal.index, jal.index)),
                                    //     self.delv().likelihood((delv, val.index)),
                                    //     self.delj().likelihood((delj, jal.index)),
                                    //     self.deld().likelihood((deld5, deld3, dal.index)),
                                    //     self.insdj().likelihood(&ins_dj_plus_last),
                                    //     self.insvd().likelihood(&ins_vd_plus_first),
                                    //     self.error().likelihood((nb_errors, length_w_del))
                                    // );

                                    if ll > 0. {
                                        result.likelihood += ll;

                                        self.vdj
                                            .dirty_update((val.index, dal.index, jal.index), ll);
                                        self.delv.dirty_update((delv, val.index), ll);
                                        self.delj.dirty_update((delj, jal.index), ll);
                                        self.deld.dirty_update((deld5, deld3, dal.index), ll);
                                        self.insdj.dirty_update(&ins_dj, first_j_nucleotide, ll);
                                        self.insvd.dirty_update(&ins_vd, last_v_nucleotide, ll);
                                        self.error.dirty_update_v_fragment(
                                            &ErrorVAlignment {
                                                val: &val,
                                                del: delv,
                                            },
                                            ll,
                                        );
                                        self.error.dirty_update_j_fragment(
                                            &ErrorJAlignment {
                                                jal: &jal,
                                                del: delj,
                                            },
                                            ll,
                                        );
                                        self.error.dirty_update_d_fragment(
                                            &ErrorDAlignment {
                                                dal: &dal,
                                                deld3,
                                                deld5,
                                            },
                                            ll,
                                        );

                                        if ip.infer_best_event && (ll > result.best_likelihood) {
                                            let event = InfEvent {
                                                v_index: val.index,
                                                v_start_gene: val.start_gene,
                                                j_index: jal.index,
                                                j_start_seq: jal.start_seq as i64
                                                    - jal.start_gene as i64,
                                                d_index: dal.index,
                                                end_v: v_end,
                                                start_d: d_start,
                                                end_d: d_end,
                                                start_j: j_start,
                                                pos_d: dal.pos,
                                                likelihood: ll,
                                                ..Default::default()
                                            };
                                            result.set_best_event(event, ip);
                                            result.best_likelihood = ll;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // update the features with the total likelihood to do the
        // averaging correctly.
        if result.likelihood > 0. {
            self.scale(result.likelihood)?;
        }

        // add a small positive likelihood to deal with the case where result.likelihood is 0.
        self.log_likelihood = Some((result.likelihood + ip.min_likelihood).log2());

        // Return the result
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn infer_given_vdj(
        &mut self,
        feature_v: &mut AggregatedFeatureEndV,
        feature_d: &mut AggregatedFeatureSpanD,
        feature_j: &mut AggregatedFeatureStartJ,
        ins_vd: &mut FeatureVD,
        ins_dj: &mut FeatureDJ,
        ip: &InferenceParameters,
        current_result: &mut ResultInference,
    ) -> Result<()> {
        let likelihood_vdj =
            self.vdj
                .likelihood((feature_v.index, feature_d.index, feature_j.index));

        let mut cutoff = ip
            .min_likelihood
            .max(ip.min_ratio_likelihood * current_result.best_likelihood);

        let (min_ev, max_ev) = (
            cmp::max(feature_v.start_v3, ins_vd.min_ev()),
            cmp::min(feature_v.end_v3, ins_vd.max_ev()),
        );
        let (min_sd, max_sd) = (
            cmp::max(feature_d.start_d5, ins_vd.min_sd()),
            cmp::min(feature_d.end_d5, ins_vd.max_sd()),
        );
        let (min_ed, max_ed) = (
            cmp::max(feature_d.start_d3, ins_dj.min_ed()),
            cmp::min(feature_d.end_d3, ins_dj.max_ed()),
        );
        let (min_sj, max_sj) = (
            cmp::max(feature_j.start_j5, ins_dj.min_sj()),
            cmp::min(feature_j.end_j5, ins_dj.max_sj()),
        );
        for ev in min_ev..max_ev {
            let likelihood_v = feature_v.likelihood(ev);
            if (likelihood_v.clone() * likelihood_vdj).max() < cutoff {
                continue;
            }
            for sd in cmp::max(ev, min_sd)..max_sd {
                let likelihood_ins_vd = ins_vd.likelihood(
                    ev,
                    sd,
                    feature_v
                        .alignment
                        .get_last_nucleotide((feature_v.end_v3 - ev - 1) as usize),
                );

                if (likelihood_v.clone() * likelihood_ins_vd.clone() * likelihood_vdj).max()
                    < cutoff
                {
                    continue;
                }
                for ed in cmp::max(sd - 1, min_ed)..max_ed {
                    let likelihood_d = feature_d.likelihood(sd, ed);
                    if (likelihood_v.clone()
                        * likelihood_ins_vd.clone()
                        * likelihood_d.clone()
                        * likelihood_vdj)
                        .max()
                        < cutoff
                    {
                        continue;
                    }
                    for sj in cmp::max(ed, min_sj)..max_sj {
                        let likelihood_ins_dj = ins_dj.likelihood(
                            ed,
                            sj,
                            feature_j
                                .alignment
                                .get_first_nucleotide((sj - feature_j.start_j5) as usize),
                        );

                        let likelihood_j = feature_j.likelihood(sj);
                        let likelihood = likelihood_v.clone()
                            * likelihood_ins_vd.clone()
                            * likelihood_d.clone()
                            * likelihood_ins_dj
                            * likelihood_j
                            * likelihood_vdj;

                        if likelihood.to_scalar() > cutoff {
                            //                         println!("{} {} {} {} {:?}", ev, sd, ed, sj, likelihood.to_scalar());
                            current_result.likelihood += likelihood.to_scalar();
                            if likelihood.max() > current_result.best_likelihood {
                                current_result.best_likelihood = likelihood.to_scalar();
                                cutoff = (ip.min_likelihood)
                                    .max(ip.min_ratio_likelihood * current_result.best_likelihood);
                                if ip.infer_best_event {
                                    // pos_d is fixed when we disaggregate D
                                    let event = InfEvent {
                                        v_index: feature_v.index,
                                        v_start_gene: feature_v.start_gene,
                                        j_index: feature_j.index,
                                        j_start_seq: feature_j.start_seq,
                                        d_index: feature_d.index,
                                        start_d: sd,
                                        end_d: ed,
                                        end_v: ev,
                                        start_j: sj,
                                        likelihood: likelihood.to_scalar(),
                                        ..Default::default()
                                    };
                                    current_result.set_best_event(event, ip);
                                }
                            }
                            if ip.infer_features.del_v {
                                feature_v.dirty_update(ev, likelihood.to_scalar());
                            }
                            if ip.infer_features.del_d {
                                feature_j.dirty_update(sj, likelihood.to_scalar());
                            }
                            if ip.infer_features.del_j {
                                feature_d.dirty_update(sd, ed, likelihood.to_scalar());
                            }
                            if ip.infer_features.ins_vd {
                                ins_vd.dirty_update(
                                    ev,
                                    sd,
                                    feature_v
                                        .alignment
                                        .get_last_nucleotide((feature_v.end_v3 - ev - 1) as usize),
                                    likelihood.to_scalar(),
                                );
                            }
                            if ip.infer_features.ins_dj {
                                ins_dj.dirty_update(
                                    ed,
                                    sj,
                                    feature_j
                                        .alignment
                                        .get_first_nucleotide((sj - feature_j.start_j5) as usize),
                                    likelihood.to_scalar(),
                                );
                            }
                            if ip.infer_features.genes {
                                self.vdj.dirty_update(
                                    (feature_v.index, feature_d.index, feature_j.index),
                                    likelihood.to_scalar(),
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn scale(&mut self, likelihood: f64) -> Result<()> {
        // Compute the new marginals for the next round
        self.vdj.scale_dirty(1. / likelihood);
        self.delv.scale_dirty(1. / likelihood);
        self.delj.scale_dirty(1. / likelihood);
        self.deld.scale_dirty(1. / likelihood);
        self.insvd.scale_dirty(1. / likelihood);
        self.insdj.scale_dirty(1. / likelihood);
        self.error.scale_dirty(1. / likelihood);
        Ok(())
    }

    pub fn normalize(&mut self) -> Result<()> {
        self.vdj = self.vdj.normalize()?;
        self.delv = self.delv.normalize()?;
        self.delj = self.delj.normalize()?;
        self.deld = self.deld.normalize()?;
        self.insvd = self.insvd.normalize()?;
        self.insdj = self.insdj.normalize()?;
        self.error = self.error.clone();
        Ok(())
    }
}
