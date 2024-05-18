#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use crate::shared::event::PyStaticEvent;
use crate::shared::StaticEvent;
use crate::shared::VJAlignment;
use crate::shared::{AlignmentParameters, ErrorParameters, Features, InferenceParameters};
use crate::shared::{Dna, Gene, ResultInference};
use crate::vdj::model::EntrySequence;
use crate::vdj::Sequence;
use crate::vdj::{display_j_alignment, display_v_alignment};
use ndarray::{Array1, Array2, Array3};

use anyhow::{anyhow, Result};

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct GenerationResult {
    pub cdr3_nt: String,
    pub cdr3_aa: Option<String>,
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
		 CDR3 (nucletides): {},\n\
		 CDR3 (amino-acids): {},\n\
		 Full sequence (nucleotides): {}...,\n\
		 V gene: {},\n\
		 J gene: {})
		 ",
            self.cdr3_nt,
            self.cdr3_aa.clone().unwrap_or("Out-of-frame".to_string()),
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
    fn get_cdr3_nt(&self) -> PyResult<String> {
        Ok(self.cdr3_nt.clone())
    }
    #[getter]
    fn get_cdr3_aa(&self) -> PyResult<Option<String>> {
        Ok(self.cdr3_aa.clone())
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Model {
    VDJ(crate::vdj::Model),
    VJ(crate::vj::Model),
}

impl Model {
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

    /// Run one round of expectation-maximization on the current model and return the next model.
    pub fn infer(
        &mut self,
        sequences: &[EntrySequence],
        features: Option<Vec<Features>>,
        alignment_params: &AlignmentParameters,
        inference_params: &InferenceParameters,
    ) -> Result<Vec<Features>> {
        match self {
            Model::VDJ(x) => x.infer(sequences, features, alignment_params, inference_params),
            Model::VJ(x) => x.infer(sequences, features, alignment_params, inference_params),
        }
    }

    /// Given a cdr3 sequence + V/J genes return a "aligned" `Sequence` object
    pub fn align_from_cdr3(
        &self,
        cdr3_seq: &Dna,
        vgenes: &Vec<Gene>,
        jgenes: &Vec<Gene>,
    ) -> Result<Sequence> {
        match self {
            Model::VDJ(x) => x.align_from_cdr3(cdr3_seq, vgenes, jgenes),
            Model::VJ(x) => x.align_from_cdr3(cdr3_seq, vgenes, jgenes),
        }
    }

    /// Align one nucleotide sequence and return a `Sequence` object
    pub fn align_sequence(
        &self,
        dna_seq: &Dna,
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
            Model::VDJ(x) => display_v_alignment(seq, v_al, &x, align_params),
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
            Model::VDJ(x) => display_j_alignment(seq, j_al, &x, align_params),
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

    pub fn set_p_del_d5_del_d3(&mut self, value: Array2<f64>) -> Result<()> {
        match self {
            Model::VDJ(x) => x.p_del_j_given_j = value,
            Model::VJ(_) => Err(anyhow!("VJ model does not have a D fragment."))?,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_markov_coefficients_vd(&self) -> Result<Array2<f64>> {
        match self {
            Model::VDJ(x) => Ok(x.markov_coefficients_vd.clone()),
            Model::VJ(_) => Err(anyhow!("VJ model does not have VD insertions.")),
        }
    }

    pub fn set_markov_coefficients_vd(&mut self, value: Array2<f64>) -> Result<()> {
        match self {
            Model::VDJ(x) => x.markov_coefficients_vd = value,
            Model::VJ(_) => Err(anyhow!("VJ model does not have VD insertions."))?,
        }
        self.initialize()?;
        Ok(())
    }

    pub fn get_markov_coefficients_dj(&self) -> Result<Array2<f64>> {
        match self {
            Model::VDJ(x) => Ok(x.markov_coefficients_dj.clone()),
            Model::VJ(_) => Err(anyhow!("VJ model does not have DJ insertions.")),
        }
    }

    pub fn set_markov_coefficients_dj(&mut self, value: Array2<f64>) -> Result<()> {
        match self {
            Model::VDJ(x) => x.markov_coefficients_dj = value,
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

    pub fn get_first_nt_bias_ins_vj(&self) -> Result<Array1<f64>> {
        match self {
            Model::VJ(x) => Ok(x.first_nt_bias_ins_vj.clone()),
            Model::VDJ(_) => Err(anyhow!("VDJ model does not have VJ insertions.")),
        }
    }

    pub fn get_first_nt_bias_ins_vd(&self) -> Result<Array1<f64>> {
        match self {
            Model::VDJ(x) => Ok(x.first_nt_bias_ins_vd.clone()),
            Model::VJ(_) => Err(anyhow!("VJ model does not have VD insertions.")),
        }
    }

    pub fn get_first_nt_bias_ins_dj(&self) -> Result<Array1<f64>> {
        match self {
            Model::VDJ(x) => Ok(x.first_nt_bias_ins_dj.clone()),
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
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct Generator {
    model: Model,
    rng: SmallRng,
}

impl Generator {
    pub fn new(
        model: Model,
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

    pub fn generate(&mut self, functional: bool) -> Result<GenerationResult> {
        self.model.generate(functional, &mut self.rng)
    }
    pub fn generate_without_errors(&mut self, functional: bool) -> GenerationResult {
        self.model
            .generate_without_errors(functional, &mut self.rng)
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
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

    /// Load the model from a set of files in IGoR format
    fn load_from_files(
        path_params: &Path,
        path_marginals: &Path,
        path_anchor_vgene: &Path,
        path_anchor_jgene: &Path,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Load the model from a set of String in IGoR format
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
    ) -> Result<Vec<Features>>;

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
        cdr3_seq: &Dna,
        vgenes: &Vec<Gene>,
        jgenes: &Vec<Gene>,
    ) -> Result<Sequence>;

    /// Align one nucleotide sequence and return a `Sequence` object
    fn align_sequence(&self, dna_seq: &Dna, align_params: &AlignmentParameters)
        -> Result<Sequence>;

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
