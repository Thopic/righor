use crate::shared::alignment::VJAlignment;
use crate::shared::entry_sequence::EntrySequence;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use crate::shared::event::PyStaticEvent;
use crate::shared::gene::Gene;
use crate::shared::gene::ModelGen;
use crate::shared::markov_chain::DNAMarkovChain;
use crate::shared::sequence::AminoAcid;
use crate::shared::sequence::Dna;
use crate::shared::sequence::Sequence;
use crate::shared::sequence::{display_j_alignment, display_v_alignment};
use crate::shared::utils::get_batches;
use crate::shared::ResultInference;
use crate::shared::StaticEvent;
use crate::shared::{AlignmentParameters, ErrorParameters, Features, InferenceParameters};
use ndarray::array;

use ndarray::{Array1, Array2, Array3};
use std::collections::HashSet;
use std::sync::Arc;

use crate::shared::errors::ErrorConstantRate;
use anyhow::{anyhow, Result};

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

use crate::shared::DnaLike;
use rand::rngs::SmallRng;
use rand::RngCore;
use rand::{Rng, SeedableRng};
use rayon::current_num_threads;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct GenerationResult {
    pub junction_nt: String,
    pub junction_aa: Option<String>,
    pub full_seq: String,
    pub v_gene: String,
    pub j_gene: String,
    pub recombination_event: StaticEvent,
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl GenerationResult {
    fn __repr__(&self) -> String {
        format!(
            "GenerationResult(\n\
		 Junction (nucletides): {},\n\
		 Junction (amino-acids): {},\n\
		 Full sequence (nucleotides): {}...,\n\
		 V gene: {},\n\
		 J gene: {})
		 ",
            self.junction_nt,
            self.junction_aa
                .clone()
                .unwrap_or("Out-of-frame".to_string()),
            &self.full_seq[0..30],
            self.v_gene,
            self.j_gene
        )
    }

    #[getter]
    fn get_recombination_event(&self) -> PyStaticEvent {
        PyStaticEvent {
            s: self.recombination_event.clone(),
        }
    }

    #[getter]
    fn get_junction_nt(&self) -> PyResult<String> {
        Ok(self.junction_nt.clone())
    }
    #[getter]
    fn get_junction_aa(&self) -> PyResult<Option<String>> {
        Ok(self.junction_aa.clone())
    }
    #[getter]
    fn get_full_seq(&self) -> PyResult<String> {
        Ok(self.full_seq.clone())
    }
    #[getter]
    fn get_v_gene(&self) -> PyResult<String> {
        Ok(self.v_gene.clone())
    }
    #[getter]
    fn get_j_gene(&self) -> PyResult<String> {
        Ok(self.j_gene.clone())
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Model {
    VDJ(crate::vdj::model::Model),
    VJ(crate::vj::model::Model),
}

impl Model {
    /// Return the core VDJ model
    pub fn get_core_vdj(&self) -> &crate::vdj::model::Model {
        match self {
            Model::VDJ(vdj) => vdj,
            Model::VJ(vj) => &vj.inner,
        }
    }

    pub fn is_productive(&self, seq: &Option<AminoAcid>) -> bool {
        match self {
            Model::VDJ(vdj) => vdj.is_productive(seq),
            Model::VJ(vj) => vj.is_productive(seq),
        }
    }

    pub fn load_from_name(
        species: &str,
        chain: &str,
        id: Option<String>,
        model_dir: &Path,
    ) -> Result<Model> {
        let result_vdj = crate::vdj::Model::load_from_name(species, chain, id.clone(), model_dir);

        if result_vdj.is_ok() {
            return Ok(Model::VDJ(result_vdj?));
        }

        let result_vj = crate::vj::Model::load_from_name(species, chain, id, model_dir);

        if result_vj.is_ok() {
            return Ok(Model::VJ(result_vj?));
        }

        // if no model can load files return all the errors
        Err(anyhow!(
            "Can't load the model.\n- VDJ try: {} \n-  VJ try: {}",
            result_vdj.err().unwrap(),
            result_vj.err().unwrap()
        ))
    }

    pub fn load_from_files(
        path_params: &Path,
        path_marginals: &Path,
        path_anchor_vgene: &Path,
        path_anchor_jgene: &Path,
    ) -> Result<Model> {
        let result_vdj = crate::vdj::Model::load_from_files(
            path_params,
            path_marginals,
            path_anchor_vgene,
            path_anchor_jgene,
        );

        if result_vdj.is_ok() {
            return Ok(Model::VDJ(result_vdj?));
        }

        let result_vj = crate::vj::Model::load_from_files(
            path_params,
            path_marginals,
            path_anchor_vgene,
            path_anchor_jgene,
        );

        if result_vj.is_ok() {
            return Ok(Model::VJ(result_vj?));
        }

        // if no model can load files return all the errors
        Err(anyhow!(
            "Can't load the model.\n- VDJ try: {} \n-  VJ try: {}",
            result_vdj.err().unwrap(),
            result_vj.err().unwrap()
        ))
    }

    /// Save the data in igor format
    pub fn save_model(&self, directory: &Path) -> Result<()> {
        match self {
            Model::VDJ(x) => x.save_model(directory),
            Model::VJ(x) => x.save_model(directory),
        }
    }

    /// Save the data in json format
    pub fn save_json(&self, filename: &Path) -> Result<()> {
        match self {
            Model::VDJ(x) => x.save_json(filename),
            Model::VJ(x) => x.save_json(filename),
        }
    }

    /// Return the genes matching a name
    pub fn genes_matching(&self, name: &str, exact: bool) -> Result<Vec<Gene>> {
        match self {
            Model::VDJ(x) => x.genes_matching(name, exact),
            Model::VJ(x) => x.genes_matching(name, exact),
        }
    }

    pub fn load_json(filename: &Path) -> Result<Model> {
        let result_vdj = crate::vdj::Model::load_json(filename);

        if result_vdj.is_ok() {
            return Ok(Model::VDJ(result_vdj?));
        }

        let result_vj = crate::vj::Model::load_json(filename);

        if result_vj.is_ok() {
            return Ok(Model::VJ(result_vj?));
        }

        // if no model can load files return all the errors
        Err(anyhow!(
            "Can't load the model.\n- VDJ try: {} \n-  VJ try: {}",
            result_vdj.err().unwrap(),
            result_vj.err().unwrap()
        ))
    }

    pub fn get_gene(&self, name: &str) -> Result<Gene> {
        for g in self.get_v_segments() {
            if g.name == name {
                return Ok(g);
            }
        }
        for g in self.get_j_segments() {
            if g.name == name {
                return Ok(g);
            }
        }
        let dgenes = self.get_d_segments();
        if dgenes.is_ok() {
            for g in dgenes.unwrap() {
                if g.name == name {
                    return Ok(g);
                }
            }
        }
        Err(anyhow!("Gene not found."))
    }

    pub fn filter_vs(&self, vs: Vec<Gene>) -> Result<Model> {
        Ok(match self {
            Model::VDJ(x) => Model::VDJ(x.filter_vs(vs)?),
            Model::VJ(x) => Model::VJ(x.filter_vs(vs)?),
        })
    }

    pub fn filter_js(&self, js: Vec<Gene>) -> Result<Model> {
        Ok(match self {
            Model::VDJ(x) => Model::VDJ(x.filter_js(js)?),
            Model::VJ(x) => Model::VJ(x.filter_js(js)?),
        })
    }

    pub fn uniform(&self) -> Result<Model> {
        Ok(match self {
            Model::VDJ(x) => Model::VDJ(x.uniform()?),
            Model::VJ(x) => Model::VJ(x.uniform()?),
        })
    }

    /// Update the internal state of the model so it stays consistent
    fn initialize(&mut self) -> Result<()> {
        match self {
            Model::VDJ(x) => x.initialize(),
            Model::VJ(x) => x.initialize(),
        }
    }

    /// Run one round of expectation-maximization on the current model
    /// Return the features and the total log-likelihood (over all sequences)
    pub fn infer(
        &mut self,
        sequences: &[EntrySequence],
        features: Option<Vec<Features>>,
        alignment_params: &AlignmentParameters,
        inference_params: &InferenceParameters,
    ) -> Result<(Vec<Features>, f64)> {
        match self {
            Model::VDJ(x) => x.infer(sequences, features, alignment_params, inference_params),
            Model::VJ(x) => x.infer(sequences, features, alignment_params, inference_params),
        }
    }

    /// Given a cdr3 sequence + V/J genes return a "aligned" `Sequence` object
    pub fn align_from_cdr3(
        &self,
        cdr3_seq: &DnaLike,
        vgenes: &[Gene],
        jgenes: &[Gene],
    ) -> Result<Sequence> {
        match self {
            Model::VDJ(x) => x.align_from_cdr3(cdr3_seq, vgenes, jgenes),
            Model::VJ(x) => x.align_from_cdr3(cdr3_seq, vgenes, jgenes),
        }
    }

    /// Align one nucleotide sequence and return a `Sequence` object
    pub fn align_sequence(
        &self,
        dna_seq: DnaLike,
        align_params: &AlignmentParameters,
    ) -> Result<Sequence> {
        match self {
            Model::VDJ(x) => x.align_sequence(dna_seq, align_params),
            Model::VJ(x) => x.align_sequence(dna_seq, align_params),
        }
    }

    /// Recreate the full sequence from the CDR3/vgene/jgene
    pub fn recreate_full_sequence(&self, dna_cdr3: &Dna, vgene: &Gene, jgene: &Gene) -> Dna {
        match self {
            Model::VDJ(x) => x.recreate_full_sequence(dna_cdr3, vgene, jgene),
            Model::VJ(x) => x.recreate_full_sequence(dna_cdr3, vgene, jgene),
        }
    }

    /// Test if self is similar to another model
    pub fn similar_to(&self, m: &Model) -> bool {
        match (&self, &m) {
            (Model::VDJ(x), Model::VDJ(y)) => x.similar_to(y.clone()),
            (Model::VJ(x), Model::VJ(y)) => x.similar_to(y.clone()),
            _ => false,
        }
    }

    /// Evaluate the sequence and return a `ResultInference` object
    pub fn evaluate(
        &self,
        sequence: EntrySequence,
        alignment_params: &AlignmentParameters,
        inference_params: &InferenceParameters,
    ) -> Result<ResultInference> {
        match self {
            Model::VDJ(x) => x.evaluate(sequence, alignment_params, inference_params),
            Model::VJ(x) => x.evaluate(sequence, alignment_params, inference_params),
        }
    }

    pub fn display_v_alignment(
        seq: &Dna,
        v_al: &VJAlignment,
        model: &Model,
        align_params: &AlignmentParameters,
    ) -> String {
        match &model {
            Model::VDJ(x) => display_v_alignment(seq, v_al, x, align_params),
            Model::VJ(x) => display_v_alignment(seq, v_al, &x.inner, align_params),
        }
    }

    pub fn display_j_alignment(
        seq: &Dna,
        j_al: &VJAlignment,
        model: &Model,
        align_params: &AlignmentParameters,
    ) -> String {
        match &model {
            Model::VDJ(x) => display_j_alignment(seq, j_al, x, align_params),
            Model::VJ(x) => display_j_alignment(seq, j_al, &x.inner, align_params),
        }
    }

    pub fn set_j_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        match self {
            Model::VDJ(x) => x.set_j_segments(value),
            Model::VJ(x) => x.set_j_segments(value),
        }
    }

    pub fn set_v_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        match self {
            Model::VDJ(x) => x.set_v_segments(value),
            Model::VJ(x) => x.set_v_segments(value),
        }
    }

    pub fn get_v_segments(&self) -> Vec<Gene> {
        match self {
            Model::VDJ(x) => x.seg_vs.clone(),
            Model::VJ(x) => x.seg_vs.clone(),
        }
    }
    pub fn get_j_segments(&self) -> Vec<Gene> {
        match self {
            Model::VDJ(x) => x.seg_js.clone(),
            Model::VJ(x) => x.seg_js.clone(),
        }
    }

    pub fn get_model_type(&self) -> ModelStructure {
        match self {
            Model::VDJ(x) => x.model_type.clone(),
            Model::VJ(x) => x.inner.model_type.clone(),
        }
    }

    pub fn set_model_type(&mut self, value: ModelStructure) -> Result<()> {
        match self {
            Model::VDJ(x) => x.model_type = value,
            Model::VJ(x) => x.inner.model_type = value,
        }
        self.initialize()
    }

    pub fn get_error(&self) -> ErrorParameters {
        match self {
            Model::VDJ(x) => x.error.clone(),
            Model::VJ(x) => x.error.clone(),
        }
    }

    pub fn set_error(&mut self, value: ErrorParameters) -> Result<()> {
        match self {
            Model::VDJ(x) => x.error = value,
            Model::VJ(x) => x.error = value,
        };
        self.initialize()
    }

    pub fn get_d_segments(&self) -> Result<Vec<Gene>> {
        match self {
            Model::VDJ(x) => Ok(x.seg_ds.clone()),
            Model::VJ(_) => Err(anyhow!("VJ Model don't have D segments")),
        }
    }

    pub fn set_d_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        match self {
            Model::VDJ(x) => x.set_d_segments(value),
            Model::VJ(_) => Err(anyhow!("VJ Model don't have D segments")),
        }
    }

    pub fn get_p_v(&self) -> Array1<f64> {
        match self {
            Model::VDJ(x) => x.p_v.clone(),
            Model::VJ(x) => x.p_v.clone(),
        }
    }

    pub fn get_p_vdj(&self) -> Result<Array3<f64>> {
        match self {
            Model::VDJ(x) => Ok(x.p_vdj.clone()),
            Model::VJ(_) => Err(anyhow!("VJ Model don't have D segments")),
        }
    }

    pub fn set_p_vdj(&mut self, value: Array3<f64>) -> Result<()> {
        match self {
            Model::VDJ(x) => x.p_vdj = value,
            Model::VJ(_) => Err(anyhow!("VJ Model don't have D segments"))?,
        }
        Ok(())
    }

    pub fn set_p_ins_vd(&mut self, value: Array1<f64>) -> Result<()> {
        match self {
            Model::VDJ(x) => x.p_ins_vd = value,
            Model::VJ(_) => Err(anyhow!("VJ Model don't have VD inserts"))?,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn set_p_ins_dj(&mut self, value: Array1<f64>) -> Result<()> {
        match self {
            Model::VDJ(x) => x.p_ins_dj = value,
            Model::VJ(_) => Err(anyhow!("VJ Model don't have DJ inserts"))?,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn set_p_ins_vj(&mut self, value: Array1<f64>) -> Result<()> {
        match self {
            Model::VDJ(_) => Err(anyhow!("VDJ Model don't have VJ inserts"))?,
            Model::VJ(x) => x.p_ins_vj = value,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_p_ins_vd(&self) -> Result<Array1<f64>> {
        Ok(match self {
            Model::VDJ(x) => x.p_ins_vd.clone(),
            Model::VJ(_) => Err(anyhow!("VJ Model don't have VD inserts"))?,
        })
    }

    pub fn get_p_ins_dj(&self) -> Result<Array1<f64>> {
        Ok(match self {
            Model::VDJ(x) => x.p_ins_dj.clone(),
            Model::VJ(_) => Err(anyhow!("VJ Model don't have DJ inserts"))?,
        })
    }

    pub fn get_p_ins_vj(&self) -> Result<Array1<f64>> {
        Ok(match self {
            Model::VJ(x) => x.p_ins_vj.clone(),
            Model::VDJ(_) => Err(anyhow!("VDJ Model don't have VJ inserts"))?,
        })
    }

    pub fn get_p_del_v_given_v(&self) -> Array2<f64> {
        match self {
            Model::VDJ(x) => x.p_del_v_given_v.clone(),
            Model::VJ(x) => x.p_del_v_given_v.clone(),
        }
    }

    pub fn set_range_del_v(&mut self, value: (i64, i64)) -> Result<()> {
        match self {
            Model::VDJ(x) => x.range_del_v = value,
            Model::VJ(x) => x.range_del_v = value,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_range_del_v(&self) -> (i64, i64) {
        match self {
            Model::VDJ(x) => x.range_del_v,
            Model::VJ(x) => x.range_del_v,
        }
    }

    pub fn set_range_del_j(&mut self, value: (i64, i64)) -> Result<()> {
        match self {
            Model::VDJ(x) => x.range_del_j = value,
            Model::VJ(x) => x.range_del_j = value,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_range_del_j(&self) -> (i64, i64) {
        match self {
            Model::VDJ(x) => x.range_del_j,
            Model::VJ(x) => x.range_del_j,
        }
    }

    pub fn set_range_del_d3(&mut self, value: (i64, i64)) -> Result<()> {
        match self {
            Model::VDJ(x) => x.range_del_d3 = value,
            Model::VJ(_) => Err(anyhow!("VJ model does not have del_d3 features."))?,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_range_del_d3(&self) -> Result<(i64, i64)> {
        match self {
            Model::VDJ(x) => Ok(x.range_del_d3),
            Model::VJ(_) => Err(anyhow!("VJ model does not have del_d3 features.")),
        }
    }

    pub fn set_range_del_d5(&mut self, value: (i64, i64)) -> Result<()> {
        match self {
            Model::VDJ(x) => x.range_del_d5 = value,
            Model::VJ(_) => Err(anyhow!("VJ model does not have del_d5 features."))?,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_range_del_d5(&self) -> Result<(i64, i64)> {
        match self {
            Model::VDJ(x) => Ok(x.range_del_d5),
            Model::VJ(_) => Err(anyhow!("VJ model does not have del_d5 features.")),
        }
    }

    pub fn set_p_del_v_given_v(&mut self, value: Array2<f64>) -> Result<()> {
        match self {
            Model::VDJ(x) => x.p_del_v_given_v = value,
            Model::VJ(x) => x.p_del_v_given_v = value,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_p_del_j_given_j(&self) -> Array2<f64> {
        match self {
            Model::VDJ(x) => x.p_del_j_given_j.clone(),
            Model::VJ(x) => x.p_del_j_given_j.clone(),
        }
    }

    pub fn set_p_del_j_given_j(&mut self, value: Array2<f64>) -> Result<()> {
        match self {
            Model::VDJ(x) => x.p_del_j_given_j = value,
            Model::VJ(x) => x.p_del_j_given_j = value,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_p_del_d5_del_d3(&self) -> Result<Array3<f64>> {
        match self {
            Model::VDJ(x) => Ok(x.p_del_d5_del_d3.clone()),
            Model::VJ(_) => Err(anyhow!("VJ model does not have a D fragment.")),
        }
    }

    pub fn set_p_del_d5_del_d3(&mut self, value: Array3<f64>) -> Result<()> {
        match self {
            Model::VDJ(x) => x.p_del_d5_del_d3 = value,
            Model::VJ(_) => Err(anyhow!("VJ model does not have a D fragment."))?,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_markov_coefficients_vd(&self) -> Result<Array2<f64>> {
        match self {
            Model::VDJ(x) => Ok(x.markov_chain_vd.transition_matrix.clone()),
            Model::VJ(_) => Err(anyhow!("VJ model does not have VD insertions.")),
        }
    }

    pub fn set_markov_coefficients_vd(&mut self, value: Array2<f64>) -> Result<()> {
        match self {
            Model::VDJ(x) => {
                x.markov_chain_vd = Arc::new(DNAMarkovChain::new(&value, false)?);
            }
            Model::VJ(_) => Err(anyhow!("VJ model does not have VD insertions."))?,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_markov_coefficients_dj(&self) -> Result<Array2<f64>> {
        match self {
            Model::VDJ(x) => Ok(x.markov_chain_dj.transition_matrix.clone()),
            Model::VJ(_) => Err(anyhow!("VJ model does not have DJ insertions.")),
        }
    }

    pub fn set_markov_coefficients_dj(&mut self, value: Array2<f64>) -> Result<()> {
        match self {
            Model::VDJ(x) => {
                x.markov_chain_dj = Arc::new(DNAMarkovChain::new(&value, true)?);
            }
            Model::VJ(_) => Err(anyhow!("VJ model does not have DJ insertions."))?,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_markov_coefficients_vj(&self) -> Result<Array2<f64>> {
        match self {
            Model::VJ(x) => Ok(x.markov_coefficients_vj.clone()),
            Model::VDJ(_) => Err(anyhow!("VDJ model does not have VJ insertions.")),
        }
    }

    pub fn set_markov_coefficients_vj(&mut self, value: Array2<f64>) -> Result<()> {
        match self {
            Model::VJ(x) => x.markov_coefficients_vj = value,
            Model::VDJ(_) => Err(anyhow!("VDJ model does not have VJ insertions."))?,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_first_nt_bias_ins_vj(&self) -> Result<Vec<f64>> {
        match self {
            Model::VJ(x) => Ok(x.get_first_nt_bias_ins_vj()?.clone()),
            Model::VDJ(_) => Err(anyhow!("VDJ model does not have VJ insertions.")),
        }
    }

    pub fn get_first_nt_bias_ins_vd(&self) -> Result<Vec<f64>> {
        match self {
            Model::VDJ(x) => Ok(x.get_first_nt_bias_ins_vd()?.clone()),
            Model::VJ(_) => Err(anyhow!("VJ model does not have VD insertions.")),
        }
    }

    pub fn get_first_nt_bias_ins_dj(&self) -> Result<Vec<f64>> {
        match self {
            Model::VDJ(x) => Ok(x.get_first_nt_bias_ins_dj()?.clone()),
            Model::VJ(_) => Err(anyhow!("VJ model does not have D genes.")),
        }
    }

    /// Return the marginal on (D, J)
    pub fn get_p_dj(&self) -> Result<Array2<f64>> {
        match self {
            Model::VDJ(x) => Ok(x.p_dj.clone()),
            Model::VJ(_) => Err(anyhow!("VJ model does not have DJ insertions.")),
        }
    }

    pub fn get_norm_productive(
        &self,
        num_monte_carlo: Option<usize>,
        conserved_j_residues: Option<&str>,
        seed: Option<u64>,
    ) -> f64 {
        let num_monte_carlo = match num_monte_carlo {
            Some(num_monte_carlo) => num_monte_carlo,
            None => 1e6 as usize,
        };
        let mut rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };
        let conserved_j_residues: HashSet<u8> = match conserved_j_residues {
            Some(conserved_j_residues) => {
                HashSet::from_iter(conserved_j_residues.bytes().into_iter())
            }
            None => HashSet::from_iter(b"FVW".into_iter().cloned()),
        };

        let num_threads = current_num_threads();
        let batches: Vec<usize> = get_batches(num_monte_carlo, num_threads);
        let seeds: Vec<u64> = (0..num_threads).map(|_| rng.next_u64()).collect();

        let mut functional_vs: HashSet<String> = HashSet::new();
        for gene in self.get_v_segments() {
            if gene.functional == "F" || gene.functional == "(F)" {
                functional_vs.insert(gene.name.to_string());
            }
        }
        let mut functional_js: HashSet<String> = HashSet::new();
        for gene in self.get_j_segments() {
            if gene.functional == "F" || gene.functional == "(F)" {
                functional_js.insert(gene.name.to_string());
            }
        }

        let num_productive: usize = seeds
            .into_par_iter()
            .enumerate()
            .map(|(idx, s)| {
                let mut child_rng = SmallRng::seed_from_u64(s);
                let mut child_model = self.clone();
                let mut count = 0;

                for _ in 0..batches[idx] {
                    let gen_result = child_model.generate_without_errors(false, &mut child_rng);
                    // Check V gene is functional.
                    if !functional_vs.contains(&gen_result.v_gene) {
                        continue;
                    }
                    // Check J gene is functional.
                    if !functional_js.contains(&gen_result.j_gene) {
                        continue;
                    }

                    let junction_aa = gen_result.junction_aa;
                    match junction_aa {
                        Some(junction_aa) => {
                            let junction_aa = junction_aa.into_bytes();

                            if junction_aa.is_empty() {
                                continue;
                            }

                            // CDR3 should have no stop codons.
                            if junction_aa.contains(&b'*') {
                                continue;
                            }

                            // CDR3 begins with a cysteine.
                            if junction_aa[0] != b'C' {
                                continue;
                            }

                            // CDR3 ends with one of the allowed conserved residues.
                            if !conserved_j_residues.contains(junction_aa.last().unwrap()) {
                                continue;
                            }
                        }
                        None => {
                            // Do not count out-of-frame sequences.
                            continue;
                        }
                    };
                    count += 1;
                }
                count
            })
            .reduce(|| 0usize, |a, b| a + b);
        (num_productive as f64) / (num_monte_carlo as f64)
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct Generator {
    model: Model,
    rng: SmallRng,
}

impl Generator {
    pub fn new(
        model: &Model,
        seed: Option<u64>,
        available_v: Option<Vec<Gene>>,
        available_j: Option<Vec<Gene>>,
    ) -> Result<Generator> {
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };

        // create an internal model in case we need to restrict the V/J genes.
        let mut internal_model = model.clone();

        if available_v.is_some() {
            internal_model = internal_model.filter_vs(available_v.unwrap())?;
        }
        if available_j.is_some() {
            internal_model = internal_model.filter_js(available_j.unwrap())?;
        }
        Ok(Generator {
            model: internal_model,
            rng,
        })
    }
}
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl Generator {
    pub fn generate(&mut self, functional: bool) -> Result<GenerationResult> {
        self.model.generate(functional, &mut self.rng)
    }
    pub fn generate_without_errors(&mut self, functional: bool) -> GenerationResult {
        self.model
            .generate_without_errors(functional, &mut self.rng)
    }

    pub fn generate_without_and_with_errors(
        &mut self,
        functional: bool,
    ) -> (GenerationResult, GenerationResult) {
        let (res_without_err, res_with_error) = self
            .model
            .generate_without_and_with_errors(functional, &mut self.rng);
        //let res_with_error = Some(res_with_error);
        (res_without_err, res_with_error.unwrap())
    }

    /// Based on `generate_many` but returns Vec<GenerationResults>
    pub fn parallel_generate(
        &mut self,
        num_monte_carlo: usize,
        functional: bool,
    ) -> Vec<GenerationResult> {
        let num_threads = current_num_threads();
        let batches: Vec<usize> = get_batches(num_monte_carlo, num_threads);
        let seeds: Vec<u64> = (0..num_threads).map(|_| self.rng.next_u64()).collect();

        seeds
            .into_par_iter()
            .enumerate()
            .flat_map_iter(|(idx, s)| {
                let mut child_generator =
                    Generator::new(&self.model, Some(s), None::<Vec<Gene>>, None::<Vec<Gene>>)
                        .unwrap();
                (0..batches[idx])
                    .into_iter()
                    .map(move |_| child_generator.generate(functional).unwrap())
            })
            .collect()
    }

    pub fn generate_many(&mut self, num_monte_carlo: usize, functional: bool) -> Vec<[String; 5]> {
        let num_threads = current_num_threads();
        let batches: Vec<usize> = get_batches(num_monte_carlo, num_threads);
        let seeds: Vec<u64> = (0..num_threads).map(|_| self.rng.next_u64()).collect();

        seeds
            .into_par_iter()
            .enumerate()
            .flat_map_iter(|(idx, s)| {
                let mut child_generator =
                    Generator::new(&self.model, Some(s), None::<Vec<Gene>>, None::<Vec<Gene>>)
                        .unwrap();
                (0..batches[idx]).into_iter().map(move |_| {
                    let gen_result = child_generator.generate(functional).unwrap();
                    [
                        gen_result.junction_aa.unwrap_or("Out-of-frame".to_string()),
                        gen_result.v_gene,
                        gen_result.j_gene,
                        gen_result.junction_nt,
                        gen_result.full_seq,
                    ]
                })
            })
            .collect::<Vec<[String; 5]>>()
    }

    pub fn generate_many_without_errors(
        &mut self,
        num_monte_carlo: usize,
        functional: bool,
    ) -> Vec<[String; 4]> {
        let num_threads = current_num_threads();
        let batches: Vec<usize> = get_batches(num_monte_carlo, num_threads);
        let seeds: Vec<u64> = (0..num_threads).map(|_| self.rng.next_u64()).collect();

        seeds
            .into_par_iter()
            .enumerate()
            .flat_map_iter(|(idx, s)| {
                let mut child_generator =
                    Generator::new(&self.model, Some(s), None::<Vec<Gene>>, None::<Vec<Gene>>)
                        .unwrap();
                (0..batches[idx]).into_iter().map(move |_| {
                    let gen_result = child_generator.generate_without_errors(functional);
                    [
                        gen_result.junction_aa.unwrap_or("Out-of-frame".to_string()),
                        gen_result.v_gene,
                        gen_result.j_gene,
                        gen_result.junction_nt,
                    ]
                })
            })
            .collect::<Vec<[String; 4]>>()
    }

    pub fn generate_many_without_and_with_errors(
        &mut self,
        num_monte_carlo: usize,
        functional: bool,
    ) -> Vec<[String; 6]> {
        let num_threads = current_num_threads();
        let batches: Vec<usize> = get_batches(num_monte_carlo, num_threads);
        let seeds: Vec<u64> = (0..num_threads).map(|_| self.rng.next_u64()).collect();

        seeds
            .into_par_iter()
            .enumerate()
            .flat_map_iter(|(idx, s)| {
                let mut child_generator =
                    Generator::new(&self.model, Some(s), None::<Vec<Gene>>, None::<Vec<Gene>>)
                        .unwrap();
                (0..batches[idx]).into_iter().map(move |_| {
                    let (res_without_error, res_with_error) =
                        child_generator.generate_without_and_with_errors(functional);
                    [
                        res_without_error
                            .junction_aa
                            .unwrap_or("Out-of-frame".to_string()),
                        res_without_error.v_gene,
                        res_without_error.j_gene,
                        res_without_error.junction_nt,
                        res_with_error
                            .junction_aa
                            .unwrap_or("Out-of-frame".to_string()),
                        res_with_error.junction_nt,
                    ]
                })
            })
            .collect::<Vec<[String; 6]>>()
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(eq, eq_int))]
#[derive(Default, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStructure {
    VDJ,
    #[default]
    VxDJ,
}

/// Generic trait to include all the models
pub trait Modelable {
    /// Load the model by looking its name in a database
    fn load_from_name(
        species: &str,
        chain: &str,
        id: Option<String>,
        model_dir: &Path,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Load the model from a set of files in `IGoR` format
    fn load_from_files(
        path_params: &Path,
        path_marginals: &Path,
        path_anchor_vgene: &Path,
        path_anchor_jgene: &Path,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Load the model from a set of String in `IGoR` format
    fn load_from_str(
        params: &str,
        marginals: &str,
        anchor_vgene: &str,
        anchor_jgene: &str,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Save the model in a given directory (write 4 files)
    fn save_model(&self, directory: &Path) -> Result<()>;

    /// Save the data in json format
    fn save_json(&self, filename: &Path) -> Result<()>
    where
        Self: Sized;

    /// Load a model saved in json format
    fn load_json(filename: &Path) -> Result<Self>
    where
        Self: Sized;

    /// Update the internal state of the model so it stays consistent
    fn initialize(&mut self) -> Result<()>;

    /// Generate a sequence
    fn generate<R: Rng>(&mut self, functional: bool, rng: &mut R) -> Result<GenerationResult>;

    /// Check if a sequence is productive
    fn is_productive(&self, seq: &Option<AminoAcid>) -> bool;

    /// Generate a sequence without and with errors
    fn generate_without_and_with_errors<R: Rng>(
        &mut self,
        functional: bool,
        rng: &mut R,
    ) -> (GenerationResult, Result<GenerationResult>);

    /// Generate a sequence without taking into account the error rate
    fn generate_without_errors<R: Rng>(
        &mut self,
        functional: bool,
        rng: &mut R,
    ) -> GenerationResult;

    /// Return an uniform model (for initializing the inference)
    fn uniform(&self) -> Result<Self>
    where
        Self: Sized;

    /// Evaluate the sequence and return a `ResultInference` object
    fn evaluate(
        &self,
        sequence: EntrySequence,
        alignment_params: &AlignmentParameters,
        inference_params: &InferenceParameters,
    ) -> Result<ResultInference>;

    /// Run one round of expectation-maximization on the current model and return the next model.
    fn infer(
        &mut self,
        sequences: &[EntrySequence],
        features: Option<Vec<Features>>,
        alignment_params: &AlignmentParameters,
        inference_params: &InferenceParameters,
    ) -> Result<(Vec<Features>, f64)>;

    // fn align_and_infer(
    //     &mut self,
    //     sequences: &[Dna],
    //     alignment_params: &AlignmentParameters,
    //     inference_params: &InferenceParameters,
    // ) -> Result<()>;

    // fn align_and_infer_from_cdr3(
    //     &mut self,
    //     sequences: &[(Dna, Vec<Gene>, Vec<Gene>)],
    //     inference_params: &InferenceParameters,
    // ) -> Result<()>;

    /// Given a cdr3 sequence + V/J genes return a "aligned" `Sequence` object
    fn align_from_cdr3(
        &self,
        cdr3_seq: &DnaLike,
        vgenes: &[Gene],
        jgenes: &[Gene],
    ) -> Result<Sequence>;

    /// Align one nucleotide sequence and return a `Sequence` object
    fn align_sequence(
        &self,
        dna_seq: DnaLike,
        align_params: &AlignmentParameters,
    ) -> Result<Sequence>;

    /// Recreate the full sequence from the CDR3/vgene/jgene
    fn recreate_full_sequence(&self, dna_cdr3: &Dna, vgene: &Gene, jgene: &Gene) -> Dna;

    /// Test if self is similar to another model
    fn similar_to(&self, m: Self) -> bool;

    /// Return the same model, with only a subset of v genes kept
    fn filter_vs(&self, vs: Vec<Gene>) -> Result<Self>
    where
        Self: Sized;
    /// Return the same model, with only a subset of j genes kept
    fn filter_js(&self, vs: Vec<Gene>) -> Result<Self>
    where
        Self: Sized;
}

impl Model {
    pub fn generate_without_errors<R: Rng>(
        &mut self,
        functional: bool,
        rng: &mut R,
    ) -> GenerationResult {
        match self {
            Model::VDJ(x) => x.generate_without_errors(functional, rng),
            Model::VJ(x) => x.generate_without_errors(functional, rng),
        }
    }

    pub fn generate<R: Rng>(&mut self, functional: bool, rng: &mut R) -> Result<GenerationResult> {
        match self {
            Model::VDJ(x) => x.generate(functional, rng),
            Model::VJ(x) => x.generate(functional, rng),
        }
    }

    pub fn generate_without_and_with_errors<R: Rng>(
        &mut self,
        functional: bool,
        rng: &mut R,
    ) -> (GenerationResult, Result<GenerationResult>) {
        match self {
            Model::VDJ(x) => x.generate_without_and_with_errors(functional, rng),
            Model::VJ(x) => x.generate_without_and_with_errors(functional, rng),
        }
    }
}

pub fn sanitize_v(genes: Vec<Gene>) -> Result<Vec<Dna>> {
    // Add palindromic inserted nucleotides to germline V sequences and cut all
    // sequences to only keep their CDR3 parts
    let mut cut_genes = Vec::<Dna>::new();
    for g in genes {
        // some V-genes are not complete. They don't appear in the model, but we
        // can't ignore them
        // TODO: I need to change the way this is done...
        if g.cdr3_pos.unwrap() >= g.seq.len() {
            cut_genes.push(Dna::new());
            continue;
        }

        let gene_seq: Dna = g
            .seq_with_pal
            .ok_or(anyhow!("Palindromic sequences not created"))?;

        let cut_gene: Dna = Dna {
            seq: gene_seq.seq[g.cdr3_pos.unwrap()..].to_vec(),
        };
        cut_genes.push(cut_gene);
    }
    Ok(cut_genes)
}

pub fn sanitize_j(genes: Vec<Gene>, max_del_j: usize) -> Result<Vec<Dna>> {
    // Add palindromic inserted nucleotides to germline J sequences and cut all
    // sequences to only keep their CDR3 parts
    let mut cut_genes = Vec::<Dna>::new();
    for g in genes {
        let gene_seq: Dna = g
            .seq_with_pal
            .ok_or(anyhow!("Palindromic sequences not created"))?;

        // for J, we want to also add the last CDR3 amino-acid (F/W)
        let cut_gene: Dna = Dna {
            seq: gene_seq.seq[..g.cdr3_pos.unwrap() + 3 + max_del_j].to_vec(),
        };
        cut_genes.push(cut_gene);
    }
    Ok(cut_genes)
}

pub fn simple_model() -> crate::vdj::Model {
    let gv1 = Gene {
        name: "V1".to_string(),
        seq: Dna::from_string("ATCTACTACTACTGCTCATGCAAAAAAAAA").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        is_functional: true,
        cdr3_pos: Some(18),
    };
    // TGCTCATGCAAAAAAGGAGGCTTTTCTCCCTGTAGTGGGAGGGAGTTAGGTGAGACACAAGGACCTCT
    // TGCTCATGCAAAAAAAAA   TTTTTCGCTTTT   GGGGGGCAGTCAGAGGAGAAACAAAGACTTAT
    let gj1 = Gene {
        name: "J1".to_string(),
        seq: Dna::from_string("GGGGGGCAGTCTTCGGAGAAACAAAGACTTAT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        is_functional: true,
        cdr3_pos: Some(11),
    };
    let gd1 = Gene {
        name: "D1".to_string(),
        seq: Dna::from_string("GGGACAGGGGGC").unwrap(), // "TTCGCTTT"
        seq_with_pal: None,
        functional: "(F)".to_string(),
        is_functional: true,
        cdr3_pos: None,
    };

    let gd2 = Gene {
        name: "D2".to_string(),
        seq: Dna::from_string("GGGACTAGCGGGGGGG").unwrap(), // TTAAAACGCAATT
        seq_with_pal: None,
        functional: "(F)".to_string(),
        is_functional: true,
        cdr3_pos: None,
    };
    let gd3 = Gene {
        name: "D2".to_string(),
        seq: Dna::from_string("GGGACTAGCGGGAGGG").unwrap(), // TAAAAACGCAATT
        seq_with_pal: None,
        functional: "(F)".to_string(),
        is_functional: true,
        cdr3_pos: None,
    };

    let mut model = crate::vdj::Model {
        seg_vs: vec![gv1],
        seg_js: vec![gj1],
        seg_ds: vec![gd1, gd2, gd3],
        p_vdj: array![[[0.6], [0.3], [0.1]]],
        p_ins_vd: array![1., 0.4, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05],
        p_ins_dj: array![1., 0.6, 0.4, 0.402, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05],
        p_del_v_given_v: array![[1.]],
        //     [0.1005],
        //     [0.2],
        //     [0.1],
        //     [0.1],
        //     [0.05],
        //     [0.1],
        //     [0.35],
        //     [0.01],
        //     [0.01],
        //     [0.01]
        // ],
        p_del_j_given_j: array![[1.]], //0.1], [0.2], [0.1], [0.1], [0.35], [0.1], [0.05],],
        p_del_d5_del_d3: array![
            [
                [0.05, 0.4, 0.2],
                [0.1, 0.1, 0.2],
                [0.05, 0.05, 0.2],
                [0.324, 0.124, 0.2],
                [0.02, 0.02, 0.2]
            ],
            [
                [0.0508, 0.2508, 0.2],
                [0.1, 0.01, 0.2],
                [0.0512, 0.0512, 0.2],
                [0.3, 0.1, 0.2],
                [0.02, 0.02, 0.2]
            ],
            [
                [0.05, 0.05, 0.2],
                [0.35, 0.15, 0.2],
                [0.05, 0.05, 0.2],
                [0.3, 0.7, 0.2],
                [0.02, 0.02, 0.2]
            ],
        ],
        markov_chain_vd: Arc::new(
            DNAMarkovChain::new(
                &array![
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.1, 0.25, 0.4],
                    [0.4, 0.25, 0.1, 0.25],
                    [0.25, 0.25, 0.25, 0.25]
                ],
                false,
            )
            .unwrap(),
        ),
        markov_chain_dj: Arc::new(
            DNAMarkovChain::new(
                &array![
                    [0.25, 0.25, 0.25, 0.25],
                    [0.2, 0.3, 0.25, 0.25],
                    [0.3, 0.2, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25]
                ],
                true,
            )
            .unwrap(),
        ),
        range_del_v: (0, 0), //(-2, 7)),
        range_del_j: (0, 0), //(-2, 4)),
        range_del_d3: (-1, 3),
        range_del_d5: (-1, 1),
        error: crate::shared::ErrorParameters::ConstantRate(ErrorConstantRate::new(0.)),
        ..Default::default()
    };

    model.initialize().unwrap();
    model
}
