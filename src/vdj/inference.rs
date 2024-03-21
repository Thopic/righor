use crate::sequence::utils::Dna;
use crate::shared::feature::*;
use crate::shared::utils::InferenceParameters;
use crate::vdj::{
    AggregatedFeatureEndV, AggregatedFeatureSpanD, AggregatedFeatureStartJ, FeatureDJ, FeatureVD,
    Model, Sequence,
};
use anyhow::{anyhow, Result};
use std::cmp;

//#[cfg(all(feature = "py_binds", feature = "pyo3"))]
//use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug, PartialEq)]
pub struct InfEvent {
    pub v_index: usize,
    pub v_start_gene: usize, // start of the sequence in the V gene
    pub j_index: usize,
    pub j_start_seq: usize, // start of the palindromic J gene (with all dels) in the sequence
    pub d_index: usize,
    // position of the v,d,j genes in the sequence
    pub end_v: i64,
    pub start_d: i64,
    pub end_d: i64,
    pub start_j: i64,

    // sequences (only added after the inference is over)
    pub ins_vd: Option<Dna>,
    pub ins_dj: Option<Dna>,
    pub d_segment: Option<Dna>,
    pub sequence: Option<Dna>,
    pub cdr3: Option<Dna>,
    pub full_sequence: Option<Dna>,
    pub reconstructed_sequence: Option<Dna>,

    // likelihood (pgen + perror)
    pub likelihood: f64,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
#[derive(Default, Clone, Debug)]
pub struct ResultInference {
    pub likelihood: f64,
    pub pgen: f64,
    best_event: Option<InfEvent>,
    best_likelihood: f64,
    pub features: Option<Features>,
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl ResultInference {
    #[getter]
    pub fn get_likelihood(&self) -> f64 {
        self.likelihood
    }
    #[getter]
    pub fn get_pgen(&self) -> f64 {
        self.pgen
    }
    #[getter]
    #[pyo3(name = "best_event")]
    pub fn py_get_best_event(&self) -> Option<InfEvent> {
        self.get_best_event()
    }
    #[getter]
    pub fn get_likelihood_best_event(&self) -> f64 {
        self.best_likelihood
    }
}

/// A Result class that's easily readable
#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all))]
pub struct ResultHuman {
    pub n_cdr3: String,
    pub aa_cdr3: String,
    pub likelihood: f64,
    pub pgen: f64,
    pub likelihood_ratio_best: f64,
    pub seq: String,
    pub full_seq: String,
    pub reconstructed_seq: String,
    pub aligned_v: String,
    pub aligned_j: String,
    pub v_name: String,
    pub j_name: String,
}

impl ResultInference {
    pub fn display(&self, model: &Model) -> Result<String> {
        if self.best_event.is_none() {
            return Ok(format!(
                "Result:\n\
		 - Likelihood: {}\n\
		 - Pgen: {}\n",
                self.likelihood, self.pgen
            ));
        }

        let rh = self.to_human(model)?;
        Ok(format!(
            "Result:\n\
	     \tLikelihood: {:.2e}, pgen: {:.2e}\n\
	     \tMost likely event:\n\
	     \t- CDR3 (nucleotides): {} \n\
	     \t- CDR3 (amino acids): {} \n\
	     \t- V name: {} \n\
	     \t- J name: {} \n\
	     \t- likelihood ratio: {} \n ",
            self.likelihood,
            self.pgen,
            rh.n_cdr3,
            rh.aa_cdr3,
            rh.v_name,
            rh.j_name,
            rh.likelihood_ratio_best
        ))
    }

    /// Translate the result to an easier to read/print version
    pub fn to_human(&self, model: &Model) -> Result<ResultHuman> {
        let best_event = self.get_best_event().ok_or(anyhow!("No event"))?;

        let translated_cdr3 = if best_event.cdr3.clone().unwrap().len() % 3 == 0 {
            best_event
                .cdr3
                .clone()
                .unwrap()
                .translate()
                .unwrap()
                .to_string()
        } else {
            String::new()
        };

        let reconstructed_seq = best_event
            .reconstructed_sequence
            .clone()
            .unwrap()
            .get_string();
        let width = reconstructed_seq.len();

        let aligned_v = format!(
            "{:width$}",
            model.seg_vs[best_event.v_index].seq.get_string(),
            width = width
        );
        let aligned_j = format!(
            "{:>width$}",
            model.seg_js[best_event.j_index].seq.get_string(),
            width = width
        );

        return Ok(ResultHuman {
            n_cdr3: best_event.cdr3.clone().unwrap().get_string(),
            aa_cdr3: translated_cdr3,
            likelihood: self.likelihood,
            pgen: self.pgen,
            likelihood_ratio_best: best_event.likelihood / self.likelihood,
            seq: best_event.sequence.clone().unwrap().get_string(),
            full_seq: best_event.full_sequence.clone().unwrap().get_string(),
            reconstructed_seq,
            aligned_v,
            aligned_j,
            v_name: model.get_v_gene(&best_event),
            j_name: model.get_j_gene(&best_event),
        });
    }

    fn impossible() -> ResultInference {
        ResultInference {
            likelihood: 0.,
            pgen: 0.,
            best_event: None,
            best_likelihood: 0.,
            features: None,
        }
    }
    pub fn set_best_event(&mut self, ev: InfEvent, ip: &InferenceParameters) {
        if ip.store_best_event {
            self.best_event = Some(ev);
        }
    }
    pub fn get_best_event(&self) -> Option<InfEvent> {
        self.best_event.clone()
    }
    /// I just store the necessary stuff in the Event variable while looping
    /// Fill event add enough to be able to completely recreate the sequence
    pub fn fill_event(&mut self, model: &Model, sequence: &Sequence) -> Result<()> {
        if !self.best_event.is_none() {
            let mut event = self.best_event.clone().unwrap();
            event.ins_vd = Some(
                sequence
                    .sequence
                    .extract_padded_subsequence(event.end_v, event.start_d),
            );
            event.ins_dj = Some(
                sequence
                    .sequence
                    .extract_padded_subsequence(event.end_d, event.start_j),
            );
            event.d_segment = Some(
                sequence
                    .sequence
                    .extract_padded_subsequence(event.start_d, event.end_d),
            );

            event.sequence = Some(sequence.sequence.clone());

            let cdr3_pos_v = model.seg_vs[event.v_index]
                .cdr3_pos
                .ok_or(anyhow!("Gene not loaded correctly"))?;
            let cdr3_pos_j = model.seg_js[event.j_index]
                .cdr3_pos
                .ok_or(anyhow!("Gene not loaded correctly"))?;

            let start_cdr3 = cdr3_pos_v as i64 - event.v_start_gene as i64;

            // careful, cdr3_pos_j does not! include the palindromic insertions
            // or the last nucleotide
            let end_cdr3 = event.j_start_seq as i64 + cdr3_pos_j as i64 - model.range_del_j.0 + 3;

            event.cdr3 = Some(
                sequence
                    .sequence
                    .extract_padded_subsequence(start_cdr3, end_cdr3),
            );

            let gene_v = model.seg_vs[event.v_index]
                .clone()
                .seq_with_pal
                .ok_or(anyhow!("Model not loaded correctly"))?;

            let gene_j = model.seg_js[event.j_index]
                .clone()
                .seq_with_pal
                .ok_or(anyhow!("Model not loaded correctly"))?;

            let mut full_seq = gene_v.extract_subsequence(0, event.v_start_gene);
            full_seq.extend(&sequence.sequence);
            full_seq.extend(
                &gene_j
                    .extract_subsequence(sequence.sequence.len() - event.j_start_seq, gene_j.len()),
            );
            event.full_sequence = Some(full_seq);

            let mut reconstructed_seq =
                gene_v.extract_subsequence(0, (event.end_v + event.v_start_gene as i64) as usize);
            reconstructed_seq.extend(&event.ins_vd.clone().unwrap());
            reconstructed_seq.extend(&event.d_segment.clone().unwrap());
            reconstructed_seq.extend(&event.ins_dj.clone().unwrap());
            reconstructed_seq.extend(&gene_j.extract_padded_subsequence(
                event.start_j - event.j_start_seq as i64,
                gene_j.len() as i64,
            ));
            event.reconstructed_sequence = Some(reconstructed_seq);
            self.best_event = Some(event);
        }
        Ok(())
    }
}

#[derive(Default, Clone, Debug)]
//#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all))]
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

            // Move the dirty proba for the next cycle, normalize everything
            self.cleanup()?;
        }

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
                                        likelihood: likelihood,
                                        ..Default::default()
                                    };
                                    current_result.set_best_event(event, ip);
                                }
                            }
                            if ip.infer {
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
