use crate::shared::feature::*;
use crate::shared::utils::{Event, InferenceParameters};
use crate::vdj::{
    AggregatedFeatureEndV, AggregatedFeatureSpanD, AggregatedFeatureStartJ, FeatureDJ, FeatureVD,
    Model, Sequence,
};
use anyhow::Result;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::{pyclass, pymethods};
use std::cmp;

// pub struct BranchCut {
//     current_likelihood: f64,
//     ip: InferenceParameters,
// }

// impl BranchCut {
//     fn new(ip: InferenceParameters) -> BranchCut {
//         BranchCut {
//             ip,
//             current_likelihood: 0.,
//         }
//     }
// }

#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all))]
pub struct ResultInference {
    pub likelihood: f64,
    pub pgen: f64,
    best_event: Option<Event>,
}

impl ResultInference {
    fn impossible() -> ResultInference {
        ResultInference {
            likelihood: 0.,
            pgen: 0.,
            best_event: None,
        }
    }

    pub fn best_event_likelihood(&self, ip: &InferenceParameters) -> f64 {
        if ip.store_best_event {
            return match &self.best_event {
                Some(e) => e.likelihood,
                None => 0.,
            };
        }
        1.
    }

    pub fn set_best_event(&mut self, ev: Event, ip: &InferenceParameters) {
        if ip.store_best_event {
            self.best_event = Some(ev);
        }
    }

    pub fn get_best_event(&self) -> Option<Event> {
        self.best_event.clone()
    }
}

#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all))]
pub struct Features {
    //    pub v: CategoricalFeature1,
    pub delv: CategoricalFeature1g1,
    // pub j: CategoricalFeature1g1,
    // pub d: CategoricalFeature1g2,
    pub vdj: CategoricalFeature3,
    pub delj: CategoricalFeature1g1,
    pub deld: CategoricalFeature2g1, // d5, d3, d
    pub insvd: InsertionFeature,
    pub insdj: InsertionFeature,
    pub error: ErrorSingleNucleotide,
}

impl Features {
    pub fn new(model: &Model) -> Result<Features> {
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

    pub fn infer(
        &mut self,
        sequence: &Sequence,
        ip: &InferenceParameters,
    ) -> Result<ResultInference> {
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
                AggregatedFeatureSpanD::new(&sequence.get_specific_dgene(d_idx), &self, ip);
            features_d.push(feature_d);
        }

        let mut features_v = Vec::new();
        for val in &sequence.v_genes {
            let feature_v = AggregatedFeatureEndV::new(val, &self, ip);
            features_v.push(feature_v);
        }

        let mut features_j = Vec::new();
        for jal in &sequence.j_genes {
            let feature_j = AggregatedFeatureStartJ::new(jal, &self, ip);
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

        // disaggregate the insertion features
        ins_vd.disaggregate(&sequence.sequence, self, ip);
        ins_dj.disaggregate(&sequence.sequence, self, ip);

        // disaggregate the v/d/j features
        for (val, v) in sequence.j_genes.iter().zip(features_v.iter_mut()) {
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

        // Move the dirty proba for the next cycle, normalize everything
        self.cleanup()?;

        // Return the result
        Ok(result)
    }

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
            if likelihood_v < ip.min_likelihood {
                continue;
            }
            for sd in cmp::max(ev, min_sd)..max_sd {
                let likelihood_ins_vd = ins_vd.likelihood(ev, sd);
                if likelihood_ins_vd * likelihood_v < ip.min_likelihood {
                    continue;
                }
                for ed in cmp::max(sd - 1, min_ed)..max_ed {
                    let likelihood_d = feature_d.likelihood(sd, ed);

                    for sj in cmp::max(ed, min_sj)..max_sj {
                        let likelihood_ins_dj = ins_dj.likelihood(ed, sj);
                        let likelihood_j = feature_j.likelihood(sj);
                        let likelihood = likelihood_v
                            * likelihood_d
                            * likelihood_j
                            * likelihood_ins_vd
                            * likelihood_ins_dj
                            * likelihood_vdj;

                        if likelihood > ip.min_likelihood {
                            current_result.likelihood += likelihood;
                            if ip.store_best_event
                                && likelihood > current_result.best_event_likelihood(ip)
                            {
                                let event = Event {
                                    v_index: feature_v.index,
                                    v_start_gene: feature_v.start_gene,
                                    j_index: feature_j.index,
                                    j_start_seq: feature_j.start_seq,
                                    d_index: Some(feature_d.index),
                                    end_v: ev,
                                    start_d: Some(sd),
                                    end_d: Some(ed),
                                    start_j: sj,
                                    likelihood: likelihood,
                                };
                                current_result.set_best_event(event, ip);
                            }
                            feature_v.dirty_update(ev, likelihood);
                            feature_j.dirty_update(sj, likelihood);
                            feature_d.dirty_update(sd, ed, likelihood);
                            ins_vd.dirty_update(ev, sd, likelihood);
                            ins_dj.dirty_update(ed, sj, likelihood);
                            self.vdj.dirty_update(
                                (feature_v.index, feature_d.index, feature_j.index),
                                likelihood,
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn cleanup(&mut self) -> Result<()> {
        // Compute the new marginals for the next round

        self.vdj = self.vdj.cleanup()?;
        self.delv = self.delv.cleanup()?;
        self.delj = self.delj.cleanup()?;
        self.deld = self.deld.cleanup()?;
        self.insvd = self.insvd.cleanup()?;
        self.insdj = self.insdj.cleanup()?;
        self.error = self.error.cleanup()?;

        Ok(())
    }
}

impl Features {
    pub fn average(features: Vec<Features>) -> Result<Features> {
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

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl Features {
    #[staticmethod]
    #[pyo3(name = "average")]
    pub fn py_average(features: Vec<Features>) -> Result<Features> {
        Features::average(features)
    }
}
