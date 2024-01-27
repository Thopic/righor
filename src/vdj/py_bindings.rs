//! Contains some of the python binding that would otherwise pollute the other files.

use crate::sequence::AminoAcid;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use crate::vdj::inference::ResultInference;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use crate::vdj::Sequence;
use crate::vdj::{Model, StaticEvent};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use crate::Dna;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use crate::{AlignmentParameters, InferenceParameters};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use anyhow::Result;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use std::path::Path;

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

use rand::rngs::SmallRng;
use rand::SeedableRng;

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct Generator {
    model: Model,
    rng: SmallRng,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub cdr3_nt: String,
    pub cdr3_aa: Option<String>,
    pub full_seq: String,
    pub v_gene: String,
    pub j_gene: String,
    pub recombination_event: StaticEvent,
}

impl Generator {
    pub fn new(model: Model, seed: Option<u64>) -> Generator {
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };

        Generator { model, rng }
    }
}

#[cfg(feature = "py_binds")]
#[pymethods]
impl Generator {
    #[new]
    pub fn py_new(
        path_params: &str,
        path_marginals: &str,
        path_v_anchors: &str,
        path_j_anchors: &str,
        seed: Option<u64>,
    ) -> Generator {
        Generator::new(
            Model::load_from_files(
                Path::new(path_params),
                Path::new(path_marginals),
                Path::new(path_v_anchors),
                Path::new(path_j_anchors),
            )
            .unwrap(),
            seed,
        )
    }

    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[pyo3(name = "generate")]
    pub fn py_generate(&mut self, functional: bool) -> GenerationResult {
        self.generate(functional)
    }
}

impl Generator {
    pub fn generate(&mut self, functional: bool) -> GenerationResult {
        let (cdr3_nt, cdr3_aa, full_sequence, event, vname, jname) =
            self.model.generate(functional, &mut self.rng);
        GenerationResult {
            full_seq: full_sequence.to_string(),
            cdr3_nt: cdr3_nt.to_string(),
            cdr3_aa: cdr3_aa.map(|x: AminoAcid| x.to_string()),
            v_gene: vname,
            j_gene: jname,
            recombination_event: event,
        }
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl Model {
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[staticmethod]
    #[pyo3(name = "load_model")]
    pub fn py_load_model(
        path_params: &str,
        path_marginals: &str,
        path_anchor_vgene: &str,
        path_anchor_jgene: &str,
    ) -> Result<Model> {
        Model::load_from_files(
            Path::new(path_params),
            Path::new(path_marginals),
            Path::new(path_anchor_vgene),
            Path::new(path_anchor_jgene),
        )
    }

    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[pyo3(name = "align_sequence")]
    pub fn py_align_sequence(
        &self,
        dna_seq: &str,
        align_params: &AlignmentParameters,
    ) -> Result<Sequence> {
        let dna = Dna::from_string(dna_seq)?;
        self.align_sequence(dna, align_params)
    }

    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[pyo3(name = "infer")]
    pub fn py_infer(
        &self,
        sequence: &Sequence,
        inference_params: &InferenceParameters,
    ) -> Result<ResultInference> {
        self.infer(sequence, inference_params)
    }

    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_v(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_v.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_v(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_v = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_dj(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_dj.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_dj(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_dj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_ins_vd(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_ins_vd.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_ins_vd(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_ins_vd = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_ins_dj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_ins_dj.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_ins_dj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_ins_dj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_del_v_given_v(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_del_v_given_v.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_del_v_given_v(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_del_v_given_v = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_del_j_given_j(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_del_j_given_j.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_del_j_given_j(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_del_j_given_j = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_del_d5_del_d3(&self, py: Python) -> Py<PyArray3<f64>> {
        self.p_del_d5_del_d3.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_del_d5_del_d3(&mut self, py: Python, value: Py<PyArray3<f64>>) -> PyResult<()> {
        self.p_del_d5_del_d3 = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_markov_coefficients_vd(&self, py: Python) -> Py<PyArray2<f64>> {
        self.markov_coefficients_vd
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_markov_coefficients_vd(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.markov_coefficients_vd = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_markov_coefficients_dj(&self, py: Python) -> Py<PyArray2<f64>> {
        self.markov_coefficients_dj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_markov_coefficients_dj(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.markov_coefficients_dj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_first_nt_bias_ins_vd(&self, py: Python) -> Py<PyArray1<f64>> {
        self.first_nt_bias_ins_vd
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_first_nt_bias_ins_vd(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.first_nt_bias_ins_vd = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_first_nt_bias_ins_dj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.first_nt_bias_ins_dj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_first_nt_bias_ins_dj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.first_nt_bias_ins_dj = value.as_ref(py).to_owned_array();
        Ok(())
    }
}
