use crate::shared::feature::*;
use crate::shared::utils::difference_as_i64;
use crate::shared::InferenceParameters;
use crate::vdj::{
    AggregatedFeatureEndV, AggregatedFeatureSpanD, AggregatedFeatureStartJ, FeatureDJ, FeatureVD,
    Model, Sequence,
};
use anyhow::Result;

use std::cmp;

#[derive(Default, Clone, Debug)]
pub struct Features {
    pub delv: CategoricalFeature1g1,
    pub vdj: CategoricalFeature3,
    pub delj: CategoricalFeature1g1,
    pub deld: CategoricalFeature2g1, // d5, d3, d
    pub insvd: InsertionFeature,
    pub insdj: InsertionFeature,
    pub error: ErrorSingleNucleotide,
}

impl FeaturesTrait for Features {
    fn generic(&self) -> FeaturesGeneric {
        FeaturesGeneric::VDJ(self.clone())
    }

    fn delv(&self) -> &CategoricalFeature1g1 {
        &self.delv
    }
    fn delj(&self) -> &CategoricalFeature1g1 {
        &self.delj
    }
    fn deld(&self) -> &CategoricalFeature2g1 {
        &self.deld
    }
    fn insvd(&self) -> &InsertionFeature {
        &self.insvd
    }
    fn insdj(&self) -> &InsertionFeature {
        &self.insdj
    }
    fn error(&self) -> &ErrorSingleNucleotide {
        &self.error
    }
    fn delv_mut(&mut self) -> &mut CategoricalFeature1g1 {
        &mut self.delv
    }
    fn delj_mut(&mut self) -> &mut CategoricalFeature1g1 {
        &mut self.delj
    }
    fn deld_mut(&mut self) -> &mut CategoricalFeature2g1 {
        &mut self.deld
    }
    fn insvd_mut(&mut self) -> &mut InsertionFeature {
        &mut self.insvd
    }
    fn insdj_mut(&mut self) -> &mut InsertionFeature {
        &mut self.insdj
    }
    fn error_mut(&mut self) -> &mut ErrorSingleNucleotide {
        &mut self.error
    }

    fn update_model(&self, model: &mut Model) -> Result<()> {
        model.p_vdj = self.vdj.probas.clone();
        model.p_del_v_given_v = self.delv.probas.clone();
        model.set_p_vdj(&self.vdj.probas.clone())?;
        model.p_del_j_given_j = self.delj.probas.clone();
        model.p_del_d5_del_d3 = self.deld.probas.clone();
        (model.p_ins_vd, model.markov_coefficients_vd) = self.insvd.get_parameters();
        (model.p_ins_dj, model.markov_coefficients_dj) = self.insdj.get_parameters();
        model.error_rate = self.error.error_rate;
        Ok(())
    }

    fn new(model: &Model) -> Result<Features> {
        Ok(Features {
            vdj: CategoricalFeature3::new(&model.p_vdj)?,
            delv: CategoricalFeature1g1::new(&model.p_del_v_given_v)?,
            delj: CategoricalFeature1g1::new(&model.p_del_j_given_j)?,
            deld: CategoricalFeature2g1::new(&model.p_del_d5_del_d3)?, // dim: (d5, d3, d)
            insvd: InsertionFeature::new(&model.p_ins_vd, &model.markov_coefficients_vd)?,
            insdj: InsertionFeature::new(&model.p_ins_dj, &model.markov_coefficients_dj)?,
            error: ErrorSingleNucleotide::new(model.error_rate)?,
        })
    }

    /// Core function, iterate over all realistic scenarios to compute the
    /// likelihood of the sequence and update the parameters
    fn infer(&mut self, sequence: &Sequence, ip: &InferenceParameters) -> Result<ResultInference> {
        // Estimate the likelihood of all possible insertions
        let mut ins_vd = match FeatureVD::new(sequence, self, ip) {
            Some(ivd) => ivd,
            None => return Ok(ResultInference::impossible()),
        };
        let mut ins_dj = match FeatureDJ::new(sequence, self, ip) {
            Some(idj) => idj,
            None => return Ok(ResultInference::impossible()),
        };

        // Define the aggregated features for this sequence:
        let mut features_d = Vec::new();
        for d_idx in 0..self.vdj.dim().1 {
            let feature_d =
                AggregatedFeatureSpanD::new(&sequence.get_specific_dgene(d_idx), self, ip);
            features_d.push(feature_d);
        }

        let mut features_v = Vec::new();
        for val in &sequence.v_genes {
            let feature_v = AggregatedFeatureEndV::new(val, self, ip);
            features_v.push(feature_v);
        }

        let mut features_j = Vec::new();
        for jal in &sequence.j_genes {
            let feature_j = AggregatedFeatureStartJ::new(jal, self, ip);
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

        if ip.infer {
            // disaggregate the insertion features
            ins_vd.disaggregate(&sequence.sequence, self, ip);
            ins_dj.disaggregate(&sequence.sequence, self, ip);

            // disaggregate the v/d/j features
            for (val, v) in sequence.v_genes.iter().zip(features_v.iter_mut()) {
                match v {
                    Some(f) => f.disaggregate(val, self, ip),
                    None => continue,
                }
            }
            for (jal, j) in sequence.j_genes.iter().zip(features_j.iter_mut()) {
                match j {
                    Some(f) => f.disaggregate(jal, self, ip),
                    None => continue,
                }
            }

            for (d_idx, d) in features_d.iter_mut().enumerate() {
                match d {
                    Some(f) => f.disaggregate(&sequence.get_specific_dgene(d_idx), self, ip),
                    None => continue,
                }
            }

            // Divide all the proba by P(R) (the probability of the sequence)
            if result.likelihood > 0. {
                self.cleanup(result.likelihood)?;
            }
        }

        // Return the result
        Ok(result)
    }

    fn average(features: Vec<Features>) -> Result<Features> {
        Ok(Features {
            vdj: CategoricalFeature3::average(features.iter().map(|a| a.vdj.clone()))?,
            delv: CategoricalFeature1g1::average(features.iter().map(|a| a.delv.clone()))?,
            delj: CategoricalFeature1g1::average(features.iter().map(|a| a.delj.clone()))?,
            deld: CategoricalFeature2g1::average(features.iter().map(|a| a.deld.clone()))?,
            insvd: InsertionFeature::average(features.iter().map(|a| a.insvd.clone()))?,
            insdj: InsertionFeature::average(features.iter().map(|a| a.insdj.clone()))?,
            error: ErrorSingleNucleotide::average(features.iter().map(|a| a.error.clone()))?,
        })
    }
}

impl Features {
    /// Brute-force inference
    /// for test-purpose only
    pub fn infer_brute_force(&mut self, sequence: &Sequence) -> Result<ResultInference> {
        let mut result = ResultInference::impossible();

        // Main loop
        for val in sequence.v_genes.clone() {
            for jal in sequence.j_genes.clone() {
                for dal in sequence.d_genes.clone() {
                    for delv in 0..self.delv.dim().0 {
                        for delj in 0..self.delj.dim().0 {
                            for deld5 in 0..self.deld().dim().0 {
                                for deld3 in 0..self.deld().dim().1 {
                                    let d_start = (dal.pos + deld5) as i64;
                                    let d_end = (dal.pos + dal.len() - deld3) as i64;
                                    let j_start = (jal.start_seq + delj) as i64;
                                    let v_end = difference_as_i64(val.end_seq, delv);
                                    if (d_start > d_end) || (j_start < d_end) || (d_start < v_end) {
                                        continue;
                                    }

                                    let mut ins_dj_plus_last =
                                        sequence.get_subsequence(d_end, j_start + 1);
                                    ins_dj_plus_last.reverse();
                                    let ins_vd_plus_first =
                                        sequence.get_subsequence(v_end - 1, d_start);

                                    let nb_errors = val.nb_errors(delv)
                                        + jal.nb_errors(delj)
                                        + dal.nb_errors(deld5, deld3);

                                    let length_w_del = val.length_with_deletion(delv)
                                        + jal.length_with_deletion(delj)
                                        + dal.length_with_deletion(deld5, deld3);

                                    let ll = self.vdj.likelihood((val.index, dal.index, jal.index))
                                        * self.delv().likelihood((delv, val.index))
                                        * self.delj().likelihood((delj, jal.index))
                                        * self.deld().likelihood((deld5, deld3, dal.index))
                                        * self.insdj().likelihood(&ins_dj_plus_last)
                                        * self.insvd().likelihood(&ins_vd_plus_first)
                                        * self.error().likelihood((nb_errors, length_w_del));

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
                                        self.delv_mut().dirty_update((delv, val.index), ll);
                                        self.delj_mut().dirty_update((delj, jal.index), ll);
                                        self.deld_mut().dirty_update((deld5, deld3, dal.index), ll);
                                        self.insdj_mut().dirty_update(&ins_dj_plus_last, ll);
                                        self.insvd_mut().dirty_update(&ins_vd_plus_first, ll);
                                        self.error_mut()
                                            .dirty_update((nb_errors, length_w_del), ll);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if result.likelihood > 0. {
            self.cleanup(result.likelihood)?;
        }

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
            if likelihood_v * likelihood_vdj < cutoff {
                continue;
            }
            for sd in cmp::max(ev, min_sd)..max_sd {
                let likelihood_ins_vd = ins_vd.likelihood(ev, sd);
                if likelihood_ins_vd * likelihood_v * likelihood_vdj < cutoff {
                    continue;
                }
                for ed in cmp::max(sd - 1, min_ed)..max_ed {
                    let likelihood_d = feature_d.likelihood(sd, ed);
                    if likelihood_ins_vd * likelihood_v * likelihood_d * likelihood_vdj < cutoff {
                        continue;
                    }

                    for sj in cmp::max(ed, min_sj)..max_sj {
                        let likelihood_ins_dj = ins_dj.likelihood(ed, sj);
                        let likelihood_j = feature_j.likelihood(sj);
                        let likelihood = likelihood_v
                            * likelihood_d
                            * likelihood_j
                            * likelihood_ins_vd
                            * likelihood_ins_dj
                            * likelihood_vdj;

                        if likelihood > cutoff {
                            current_result.likelihood += likelihood;
                            if likelihood > current_result.best_likelihood {
                                current_result.best_likelihood = likelihood;
                                cutoff = (ip.min_likelihood)
                                    .max(ip.min_ratio_likelihood * current_result.best_likelihood);
                                if ip.store_best_event {
                                    let event = InfEvent {
                                        v_index: feature_v.index,
                                        v_start_gene: feature_v.start_gene,
                                        j_index: feature_j.index,
                                        j_start_seq: feature_j.start_seq,
                                        d_index: feature_d.index,
                                        end_v: ev,
                                        start_d: sd,
                                        end_d: ed,
                                        start_j: sj,
                                        likelihood,
                                        ..Default::default()
                                    };
                                    current_result.set_best_event(event, ip);
                                }
                            }
                            if ip.infer {
                                if ip.infer_genes {
                                    feature_v.dirty_update(ev, likelihood);
                                    feature_j.dirty_update(sj, likelihood);
                                    feature_d.dirty_update(sd, ed, likelihood);
                                }
                                if ip.infer_insertions {
                                    ins_vd.dirty_update(ev, sd, likelihood);
                                    ins_dj.dirty_update(ed, sj, likelihood);
                                }
                                self.vdj.dirty_update(
                                    (feature_v.index, feature_d.index, feature_j.index),
                                    likelihood,
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn cleanup(&mut self, likelihood: f64) -> Result<()> {
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
}

impl Features {
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
