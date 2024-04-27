use crate::shared::feature::*;
use crate::shared::{FeaturesGeneric, FeaturesTrait, InferenceParameters, ResultInference};
use crate::v_dj::AggregatedFeatureStartDAndJ;
use crate::vdj::{
    AggregatedFeatureEndV, AggregatedFeatureSpanD, AggregatedFeatureStartJ, FeatureDJ, FeatureVD,
    Model, Sequence,
};
use anyhow::Result;
use ndarray::Axis;
use std::cmp;

#[derive(Default, Clone, Debug)]
pub struct Features {
    pub delv: CategoricalFeature1g1,
    pub d: CategoricalFeature1g1, // d given j
    pub vj: CategoricalFeature2,  // v, j
    pub delj: CategoricalFeature1g1,
    pub deld: CategoricalFeature2g1, // d5, d3, d
    pub insvd: InsertionFeature,
    pub insdj: InsertionFeature,
    pub error: ErrorSingleNucleotide,
}

impl FeaturesTrait for Features {
    fn generic(&self) -> FeaturesGeneric {
        FeaturesGeneric::VxDJ(self.clone())
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
        let pvj = self.vj.probas.clone();
        let pd_given_j = self.d.probas.clone();
        model.p_vdj = pvj.insert_axis(Axis(1)) * pd_given_j.insert_axis(Axis(0));
        model.p_del_v_given_v = self.delv.probas.clone();
        model.set_p_vdj(&model.p_vdj.clone())?;
        model.p_del_j_given_j = self.delj.probas.clone();
        model.p_del_d5_del_d3 = self.deld.probas.clone();
        (model.p_ins_vd, model.markov_coefficients_vd) = self.insvd.get_parameters();
        (model.p_ins_dj, model.markov_coefficients_dj) = self.insdj.get_parameters();
        model.error_rate = self.error.error_rate;
        Ok(())
    }

    fn new(model: &Model) -> Result<Features> {
        Ok(Features {
            vj: CategoricalFeature2::new(&model.get_p_vj())?,
            d: CategoricalFeature1g1::new(&model.get_p_d_given_j())?,
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
        let mut agg_ins_vd = match FeatureVD::new(sequence, self, ip) {
            Some(ivd) => ivd,
            None => return Ok(ResultInference::impossible()),
        };
        let mut agg_ins_dj = match FeatureDJ::new(sequence, self, ip) {
            Some(idj) => idj,
            None => return Ok(ResultInference::impossible()),
        };

        // Define the aggregated features for this sequence:
        let mut features_v = Vec::new();
        for val in &sequence.v_genes {
            let feature_v = AggregatedFeatureEndV::new(val, self, ip);
            features_v.push(feature_v);
        }

        let mut features_j = Vec::new();
        for jal in &sequence.j_genes {
            let feature_j = AggregatedFeatureStartJ::new(jal, self, ip);
            if let Some(feat_j) = feature_j {
                features_j.push(feat_j);
            }
        }

        let mut features_d = Vec::new();
        for d_idx in 0..self.d.dim().0 {
            let feature_d =
                AggregatedFeatureSpanD::new(&sequence.get_specific_dgene(d_idx), self, ip);
            if let Some(feat_d) = feature_d {
                features_d.push(feat_d);
            }
        }

        if features_d.is_empty() {
            println!("This probably shouldn't happen...");
            return Ok(ResultInference::impossible());
        }

        let mut features_dj = Vec::new();
        for feature_j in &features_j {
            let feature_dj =
                AggregatedFeatureStartDAndJ::new(feature_j, &features_d, &agg_ins_dj, self, ip);
            features_dj.push(feature_dj);
        }

        let mut result = ResultInference::impossible();

        // Main loop
        for v in features_v.iter_mut().filter_map(|x| x.as_mut()) {
            for dj in features_dj.iter_mut().filter_map(|x| x.as_mut()) {
                self.infer_given_vdj(v, dj, &mut agg_ins_vd, ip, &mut result)?;
            }
        }

        if ip.infer {
            // disaggregate the v/dj features
            for (val, v) in sequence.v_genes.iter().zip(features_v.iter_mut()) {
                match v {
                    Some(f) => f.disaggregate(val, self, ip),
                    None => continue,
                }
            }
            for (feature_j, dj) in features_j.iter_mut().zip(features_dj.iter_mut()) {
                match dj {
                    Some(f) => {
                        f.disaggregate(feature_j, &mut features_d, &mut agg_ins_dj, self, ip)
                    }
                    None => continue,
                }
            }

            for (jal, j) in sequence.j_genes.iter().zip(features_j.iter_mut()) {
                j.disaggregate(jal, self, ip)
            }

            for (d_idx, d) in features_d.iter_mut().enumerate() {
                d.disaggregate(&sequence.get_specific_dgene(d_idx), self, ip);
            }

            // disaggregate the insertion features
            agg_ins_vd.disaggregate(&sequence.sequence, self, ip);
            agg_ins_dj.disaggregate(&sequence.sequence, self, ip);
        }
        if result.likelihood > 0. {
            self.cleanup(result.likelihood)?;
        }
        // Return the result
        Ok(result)
    }

    fn average(features: Vec<Features>) -> Result<Features> {
        Ok(Features {
            vj: CategoricalFeature2::average(features.iter().map(|a| a.vj.clone()))?,
            d: CategoricalFeature1g1::average(features.iter().map(|a| a.d.clone()))?,
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
    #[allow(clippy::too_many_arguments)]
    pub fn infer_given_vdj(
        &mut self,
        feature_v: &mut AggregatedFeatureEndV,
        feature_dj: &mut AggregatedFeatureStartDAndJ,
        ins_vd: &mut FeatureVD,
        ip: &InferenceParameters,
        current_result: &mut ResultInference,
    ) -> Result<()> {
        let likelihood_vj = self.vj.likelihood((feature_v.index, feature_dj.index));

        let mut cutoff = ip
            .min_likelihood
            .max(ip.min_ratio_likelihood * current_result.best_likelihood);

        let (min_ev, max_ev) = (
            cmp::max(feature_v.start_v3, ins_vd.min_ev()),
            cmp::min(feature_v.end_v3, ins_vd.max_ev()),
        );
        let (min_sd, max_sd) = (
            cmp::max(feature_dj.start_d5, ins_vd.min_sd()),
            cmp::min(feature_dj.end_d5, ins_vd.max_sd()),
        );

        for ev in min_ev..max_ev {
            let likelihood_v = feature_v.likelihood(ev);
            if likelihood_v * likelihood_vj < cutoff {
                continue;
            }
            for sd in cmp::max(ev, min_sd)..max_sd {
                let likelihood_ins_vd = ins_vd.likelihood(ev, sd);
                let likelihood_dj = feature_dj.likelihood(sd);
                let likelihood = likelihood_v * likelihood_ins_vd * likelihood_dj * likelihood_vj;

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
                                j_index: feature_dj.index,
                                j_start_seq: feature_dj.start_seq,
                                d_index: feature_dj.most_likely_d_index,
                                end_v: ev,
                                start_d: sd,
                                end_d: feature_dj.most_likely_d_end,
                                start_j: feature_dj.most_likely_j_start,
                                likelihood,
                                ..Default::default()
                            };
                            current_result.set_best_event(event, ip);
                        }
                    }
                    if ip.infer {
                        if ip.infer_genes {
                            feature_v.dirty_update(ev, likelihood);
                            feature_dj.dirty_update(sd, likelihood);
                        }
                        if ip.infer_insertions {
                            ins_vd.dirty_update(ev, sd, likelihood);
                        }
                        self.vj
                            .dirty_update((feature_v.index, feature_dj.index), likelihood);
                    }
                }
            }
        }

        Ok(())
    }

    pub fn cleanup(&mut self, likelihood: f64) -> Result<()> {
        // Compute the new marginals for the next round
        self.vj.scale_dirty(1. / likelihood);
        self.d.scale_dirty(1. / likelihood);
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
        self.vj = self.vj.normalize()?;
        self.d = self.d.normalize()?;
        self.delv = self.delv.normalize()?;
        self.delj = self.delj.normalize()?;
        self.deld = self.deld.normalize()?;
        self.insvd = self.insvd.normalize()?;
        self.insdj = self.insdj.normalize()?;
        self.error = self.error.clone();
        Ok(())
    }
}
