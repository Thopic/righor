use crate::shared::feature::Feature;
use crate::shared::InfEvent;
use crate::shared::{errors::FeatureError, InferenceParameters, ResultInference};
use crate::shared::{
    CategoricalFeature1g1, CategoricalFeature2, CategoricalFeature2g1, DNAMarkovChain,
    ErrorParameters, InsertionFeature,
};
use crate::v_dj::AggregatedFeatureStartDAndJ;
use crate::vdj::{
    AggregatedFeatureEndV, AggregatedFeatureSpanD, FeatureDJ, FeatureVD, Model, Sequence,
};
use anyhow::Result;
use ndarray::Axis;
use std::cmp;
use std::sync::Arc;

#[derive(Default, Clone, Debug)]
pub struct Features {
    pub delv: CategoricalFeature1g1,
    pub d: CategoricalFeature1g1, // d given j
    pub vj: CategoricalFeature2,  // v, j
    pub delj: CategoricalFeature1g1,
    pub deld: CategoricalFeature2g1, // d5, d3, d
    pub insvd: InsertionFeature,
    pub insdj: InsertionFeature,
    pub error: FeatureError,
}

impl Features {
    /// Update the model from a vector of features and return an updated vector of features
    pub fn update(features: Vec<Features>, model: &mut Model) -> Result<Vec<Features>> {
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
        let vj = CategoricalFeature2::average(features.iter().map(|a| a.vj.clone()))?;
        let d_given_j = CategoricalFeature1g1::average(features.iter().map(|a| a.d.clone()))?;
        let p_vdj =
            vj.clone().probas.insert_axis(Axis(1)) * d_given_j.clone().probas.insert_axis(Axis(0));

        model.set_p_vdj(&p_vdj)?;
        model.p_del_v_given_v = delv.clone().probas;
        model.p_del_j_given_j = delj.clone().probas;
        model.p_del_d5_del_d3 = deld.clone().probas;

        (model.p_ins_vd, model.markov_coefficients_vd) = insvd.get_parameters();
        (model.p_ins_dj, model.markov_coefficients_dj) = insdj.get_parameters();

        // Now update the features vector
        let mut new_features = Vec::new();
        for error in errors {
            new_features.push(Features {
                vj: vj.clone(),
                d: d_given_j.clone(),
                delv: delv.clone(),
                delj: delj.clone(),
                deld: deld.clone(),
                insvd: insvd.correct_for_error(&error).clone(),
                insdj: insdj.correct_for_error(&error).clone(),
                error: error.clone(),
            })
        }

        Ok(new_features)
    }

    pub fn new(model: &Model) -> Result<Features> {
        Ok(Features {
            vj: CategoricalFeature2::new(&model.get_p_vj())?,
            d: CategoricalFeature1g1::new(&model.get_p_d_given_j())?,
            delv: CategoricalFeature1g1::new(&model.p_del_v_given_v)?,
            delj: CategoricalFeature1g1::new(&model.p_del_j_given_j)?,
            deld: CategoricalFeature2g1::new(&model.p_del_d5_del_d3)?, // dim: (d5, d3, d)
            insvd: InsertionFeature::new(
                &model.p_ins_vd,
                Arc::new(DNAMarkovChain::new(&model.markov_coefficients_vd)?),
            )?,
            insdj: InsertionFeature::new(
                &model.p_ins_dj,
                Arc::new(DNAMarkovChain::new(&model.markov_coefficients_dj)?),
            )?,
            error: model.error.get_feature()?,
        })
    }

    /// Core function, iterate over all realistic scenarios to compute the
    /// likelihood of the sequence and update the parameters
    pub fn infer(
        &mut self,
        sequence: &Sequence,
        ip: &InferenceParameters,
    ) -> Result<ResultInference> {
        // Estimate the likelihood of all possible insertions
        let mut agg_ins_vd = match FeatureVD::new(
            sequence,
            &self.insvd,
            self.delv.dim().0,
            self.deld.dim().0,
            ip,
        ) {
            Some(ivd) => ivd,
            None => return Ok(ResultInference::impossible()),
        };
        let mut agg_ins_dj = match FeatureDJ::new(
            sequence,
            &self.insdj,
            self.deld.dim().1,
            self.delj.dim().0,
            ip,
        ) {
            Some(idj) => idj,
            None => return Ok(ResultInference::impossible()),
        };

        // Define the aggregated features for this sequence:
        let mut features_v = Vec::new();
        for val in &sequence.v_genes {
            let feature_v = AggregatedFeatureEndV::new(val, &self.delv, &self.error, ip);
            features_v.push(feature_v);
        }

        let mut features_d = Vec::new();
        for d_idx in 0..self.d.dim().0 {
            let feature_d = AggregatedFeatureSpanD::new(
                &sequence.get_specific_dgene(d_idx),
                &self.deld,
                &self.error,
                ip,
            );
            if feature_d.is_some() {
                features_d.push(feature_d.unwrap());
            }
        }

        if features_d.is_empty() {
            println!("This probably shouldn't happen...");
            return Ok(ResultInference::impossible());
        }

        let mut features_dj = Vec::new();
        for jal in &sequence.j_genes {
            let feature_dj =
                AggregatedFeatureStartDAndJ::new(jal, &features_d, &agg_ins_dj, self, ip);
            features_dj.push(feature_dj);
        }

        let mut result = ResultInference::impossible();

        // Main loop
        for v in features_v.iter_mut().filter_map(|x| x.as_mut()) {
            for dj in features_dj.iter_mut().filter_map(|x| x.as_mut()) {
                self.infer_given_vdj(v, dj, &mut agg_ins_vd, ip, &mut result)?;
            }
        }

        if ip.infer_features {
            // disaggregate the v/dj features
            for (val, v) in sequence.v_genes.iter().zip(features_v.iter_mut()) {
                match v {
                    Some(f) => f.disaggregate(val, &mut self.delv, &mut self.error, ip),
                    None => continue,
                }
            }
            for (jal, dj) in sequence.j_genes.iter().zip(features_dj.iter_mut()) {
                match dj {
                    Some(f) => f.disaggregate(
                        jal,
                        &mut features_d,
                        &mut agg_ins_dj,
                        self,
                        &mut result.best_event,
                        ip,
                    ),
                    None => continue,
                }
            }
            for (d_idx, d) in features_d.iter_mut().enumerate() {
                d.disaggregate(
                    &sequence.get_specific_dgene(d_idx),
                    &mut self.deld,
                    &mut self.error,
                    &mut result.best_event,
                    ip,
                );
            }

            // disaggregate the insertion features
            agg_ins_vd.disaggregate(&sequence.sequence, &mut self.insvd, ip);
            agg_ins_dj.disaggregate(&sequence.sequence, &mut self.insdj, ip);
        }
        if result.likelihood > 0. {
            self.cleanup(result.likelihood)?;
        }
        // Return the result
        Ok(result)
    }

    //    pub fn average(features: Vec<Features>) -> Result<Features> {}
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
        let likelihood_vj = self.vj.likelihood((feature_v.index, feature_dj.j_index()));

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
                            // We just set the ones we have right now
                            let event = InfEvent {
                                v_index: feature_v.index,
                                v_start_gene: feature_v.start_gene,
                                j_start_seq: feature_dj.j_start_seq(),
                                j_index: feature_dj.j_index(),
                                end_v: ev,
                                start_d: sd,
                                likelihood,
                                ..Default::default()
                            };
                            current_result.set_best_event(event, ip);
                        }
                    }
                    if ip.infer_features {
                        feature_v.dirty_update(ev, likelihood);
                        feature_dj.dirty_update(sd, likelihood);
                        ins_vd.dirty_update(ev, sd, likelihood);
                        self.vj
                            .dirty_update((feature_v.index, feature_dj.j_index()), likelihood);
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
