use crate::sequence::VJAlignment;
use crate::shared::feature::*;
use crate::shared::utils::InferenceParameters;
use crate::vdj::{
    AggregatedFeatureEndV, AggregatedFeatureSpanD, AggregatedFeatureStartJ, FeatureDJ, FeatureVD,
    Model, Sequence,
};
use anyhow::Result;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::{pyclass, pymethods};
use std::cmp;

#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all))]
pub struct Features {
    pub v: CategoricalFeature1,
    pub delv: CategoricalFeature1g1,
    pub j: CategoricalFeature1g1,
    pub d: CategoricalFeature1g2,
    pub delj: CategoricalFeature1g1,
    pub deld: CategoricalFeature2g1, // d3, d5, d
    pub insvd: InsertionFeature,
    pub insdj: InsertionFeature,
    pub error: ErrorSingleNucleotide,
}

impl Features {
    pub fn new(model: &Model) -> Result<Features> {
        Ok(Features {
            v: CategoricalFeature1::new(&model.p_v)?,
            delv: CategoricalFeature1g1::new(&model.p_del_v_given_v)?,
            j: CategoricalFeature1g1::new(&model.p_j_given_v)?,
            d: CategoricalFeature1g2::new(&model.p_d_given_vj)?,
            delj: CategoricalFeature1g1::new(&model.p_del_j_given_j)?,
            deld: CategoricalFeature2g1::new(&model.p_del_d3_del_d5)?, // dim: (d3, d5, d)
            insvd: InsertionFeature::new(&model.p_ins_vd, &model.markov_coefficients_vd)?,
            insdj: InsertionFeature::new(&model.p_ins_dj, &model.markov_coefficients_dj)?,
            error: ErrorSingleNucleotide::new(model.error_rate)?,
        })
    }

    // TODO: needs to make a structure "InferenceResult"
    pub fn infer(&mut self, sequence: &Sequence, ip: &InferenceParameters) -> Result<f64> {
        // Estimate the likelihood of all possible insertions
        let mut ins_vd = FeatureVD::new(sequence, self, ip);
        let mut ins_dj = FeatureDJ::new(sequence, self, ip);

        // Now start the inference
        let mut l_total = 0.;
        for val in &sequence.v_genes {
            for jal in &sequence.j_genes {
                l_total +=
                    self.infer_given_alignment(sequence, &val, &jal, &mut ins_vd, &mut ins_dj, ip)?;
            }
        }

        // disaggregate the insertion features
        ins_vd.disaggregate(&sequence.sequence, self, ip);
        ins_dj.disaggregate(&sequence.sequence, self, ip);
        // Move the dirty proba for the next cycle, normalize everything
        self.cleanup()?;
        Ok(l_total)
    }

    pub fn infer_given_alignment(
        &mut self,
        sequence: &Sequence,
        val: &VJAlignment,
        jal: &VJAlignment,
        ins_vd: &mut FeatureVD,
        ins_dj: &mut FeatureDJ,
        ip: &InferenceParameters,
    ) -> Result<f64> {
        let mut feature_v = match AggregatedFeatureEndV::new(val, &self, ip) {
            Some(f) => f,
            None => return Ok(0.),
        };
        let mut feature_d = AggregatedFeatureSpanD::new(val, &sequence.d_genes, jal, &self, ip);
        let mut feature_j = match AggregatedFeatureStartJ::new(val, jal, &self, ip) {
            Some(f) => f,
            None => return Ok(0.),
        };

        let mut l_total = 0.;
        for ev in cmp::max(feature_v.start_v3, ins_vd.min_ev())
            ..cmp::min(feature_v.end_v3, ins_vd.max_ev())
        {
            let log_likelihood_v = feature_v.log_likelihood(ev);
            if log_likelihood_v < ip.min_log_likelihood {
                continue;
            }
            for sd in ev.max(feature_d.start_d5).max(ins_vd.min_sd())
                ..feature_d.end_d5.min(ins_vd.max_sd())
            {
                let log_likelihood_ins_vd = ins_vd.log_likelihood(ev, sd);
                if log_likelihood_ins_vd + log_likelihood_v < ip.min_log_likelihood {
                    continue;
                }
                for ed in (sd - 1).max(feature_d.start_d3).max(ins_dj.min_ed())
                    ..feature_d.end_d3.min(ins_dj.max_ed())
                {
                    for sj in ed.max(feature_j.start_j5).max(ins_dj.min_sj())
                        ..feature_j.end_j5.min(ins_dj.max_sj())
                    {
                        let log_likelihood = log_likelihood_v
                            + feature_j.log_likelihood(sj)
                            + feature_d.log_likelihood(sd, ed)
                            + log_likelihood_ins_vd
                            + ins_dj.log_likelihood(ed, sj);

                        if log_likelihood > ip.min_log_likelihood {
                            let likelihood = log_likelihood.exp2();
                            l_total += likelihood;
                            feature_v.dirty_update(ev, likelihood);
                            feature_j.dirty_update(sj, likelihood);
                            feature_d.dirty_update(sd, ed, likelihood);
                            ins_vd.dirty_update(ev, sd, likelihood);
                            ins_dj.dirty_update(ed, sj, likelihood);
                        }
                    }
                }
            }
        }

        // Transform back the aggregated feature into marginals
        if l_total != 0. {
            feature_v.disaggregate(val, self, ip);
            feature_d.disaggregate(val, &sequence.d_genes, jal, self, ip);
            feature_j.disaggregate(val, jal, self, ip);
        }

        Ok(l_total)
    }

    // pub fn infer(&mut self, sequence: &Sequence, ip: &InferenceParameters) -> Result<f64> {
    //     // let mut probability_generation: f64 = 0.; // need to deal with that
    //     // let mut probability_generation_no_error: f64 = 0.; // TODO return this too
    //     // let mut best_events = Vec::<(f64, StaticEvent)>::new();

    //     let mut feature_v = AggregatedFeatureEndV::new(sequence, &self, ip);
    //     let mut feature_dj = AggregatedFeatureDJ::new(sequence, &self, ip);

    //     let mut ll_ins_vd = Vec::new();
    //     for ev in feature_v.start_v3..feature_v.end_v3 {
    //         for sd in cmp::max(ev, feature_dj.start_d5)..feature_dj.end_d5 {
    //             if sd - ev <= self.insvd.max_nb_insertions() as i64 {
    //                 let ins_vd_plus_first = sequence.get_subsequence(ev - 1, sd);
    //                 ll_ins_vd.push(((ev, sd), self.insvd.log_likelihood(&ins_vd_plus_first)));
    //             }
    //         }
    //     }
    //     let log_likelihood_ins_vd = RangeArray2::new(&ll_ins_vd);

    //     let mut ll_ins_dj = Vec::new();
    //     for sj in feature_dj.start_j5..feature_dj.end_j5 {
    //         for ed in feature_dj.start_d3..cmp::min(feature_dj.end_d3, sj + 1) {
    //             if sj - ed <= self.insdj.max_nb_insertions() as i64 {
    //                 // careful we need to reverse ins_dj for the inference
    //                 let mut ins_dj_plus_last = sequence.get_subsequence(ed, sj + 1);
    //                 ins_dj_plus_last.reverse();
    //                 ll_ins_dj.push(((ed, sj), self.insdj.log_likelihood(&ins_dj_plus_last)));
    //             }
    //         }
    //     }

    //     let log_likelihood_ins_dj = RangeArray2::new(&ll_ins_dj);

    //     let mut l_total = 0.;

    //     for ev in feature_v.start_v3..feature_v.end_v3 {
    //         for sd in cmp::max(ev, feature_dj.start_d5)..feature_dj.end_d5 {
    //             if sd - ev > self.insvd.max_nb_insertions() as i64 {
    //                 continue;
    //             }
    //             let mut likelihood_v = 0.;
    //             let ins_vd_plus_first = sequence.get_subsequence(ev - 1, sd);
    //             let log_likelihood_v = feature_v.log_likelihood(ev);
    //             if log_likelihood_ins_vd.get((ev, sd)) < ip.min_log_likelihood {
    //                 continue;
    //             }
    //             for ed in cmp::max(sd - 1, feature_dj.start_d3)..feature_dj.end_d3 {
    //                 for sj in cmp::max(ed, feature_dj.start_j5)..feature_dj.end_j5 {
    //                     if sj - ed > self.insdj.max_nb_insertions() as i64 {
    //                         continue;
    //                     }
    //                     let mut likelihood_ins = 0.;
    //                     let mut ins_dj_plus_last = sequence.get_subsequence(ed, sj + 1);
    //                     ins_dj_plus_last.reverse();
    //                     let log_likelihood_ins = log_likelihood_ins_vd.get((ev, sd))
    //                         + log_likelihood_ins_dj.get((ed, sj));
    //                     for j_idx in 0..feature_dj.nb_j_alignments {
    //                         let log_likelihood = feature_dj.log_likelihood(sd, ed, sj, j_idx)
    //                             + log_likelihood_ins
    //                             + log_likelihood_v;

    //                         if log_likelihood > ip.min_log_likelihood {
    //                             let likelihood = log_likelihood.exp2();
    //                             likelihood_v += likelihood;
    //                             likelihood_ins += likelihood;
    //                             l_total += likelihood;
    //                             feature_dj.dirty_update(sd, ed, sj, j_idx, likelihood);
    //                         }
    //                     }
    //                     if likelihood_ins > 0. {
    //                         self.insvd.dirty_update(&ins_vd_plus_first, likelihood_ins);
    //                         self.insdj.dirty_update(&ins_dj_plus_last, likelihood_ins);
    //                     }
    //                 }
    //             }
    //             if likelihood_v > 0. {
    //                 feature_v.dirty_update(ev, likelihood_v);
    //             }
    //         }
    //     }

    //     if l_total != 0. {
    //         feature_v.cleanup(&sequence, self, ip);
    //         feature_dj.cleanup(&sequence, self, ip);
    //         *self = self.cleanup()?;
    //     }

    //     Ok(l_total)
    // }

    pub fn cleanup(&mut self) -> Result<()> {
        // Compute the new marginals for the next round

        self.v = self.v.cleanup()?;
        self.d = self.d.cleanup()?;
        self.j = self.j.cleanup()?;
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
            v: CategoricalFeature1::average(features.iter().map(|a| a.v.clone()))?,
            delv: CategoricalFeature1g1::average(features.iter().map(|a| a.delv.clone()))?,
            j: CategoricalFeature1g1::average(features.iter().map(|a| a.j.clone()))?,
            d: CategoricalFeature1g2::average(features.iter().map(|a| a.d.clone()))?,
            delj: CategoricalFeature1g1::average(features.iter().map(|a| a.delj.clone()))?,
            deld: CategoricalFeature2g1::average(features.iter().map(|a| a.deld.clone()))?,
            insvd: InsertionFeature::average(features.iter().map(|a| a.insvd.clone()))?,
            insdj: InsertionFeature::average(features.iter().map(|a| a.insdj.clone()))?,
            error: ErrorSingleNucleotide::average(features.iter().map(|a| a.error.clone()))?,
        })
    }

    pub fn normalize(&mut self) -> Result<()> {
        self.v = self.v.normalize()?;
        self.d = self.d.normalize()?;
        self.j = self.j.normalize()?;
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
