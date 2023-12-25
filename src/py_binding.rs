use crate::feature::{
    CategoricalFeature1, CategoricalFeature1g1, CategoricalFeature2, CategoricalFeature2g1,
    MarkovFeature,
};

use crate::model::{ModelVDJ, ModelVJ};
use crate::utils_sequences::AminoAcid;
use crate::FeaturesVDJ;
use anyhow::Result;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::path::Path;

// Alignment utilities ////

// Inference utilities ////

// #[pyfunction]
// pub fn expectation_maximization_vdj(
//     sequences: Vec<SequenceVDJ>,
//     model: &ModelVDJ,
//     inference_params: &InferenceParameters,
// ) -> Result<ModelVDJ> {
//     let margs = FeaturesVDJ::new(model, inference_params)?;
//     let mut new_model = model.clone();
//     let new_margs = expectation_maximization(&margs, &sequences, inference_params)?;
//     new_margs.update_model(&mut new_model);
//     Ok(new_model)
// }

// Generation utilities ////

#[pyclass]
#[derive(Debug, Clone)]
pub struct GenerationResult {
    #[pyo3(get, set)]
    pub cdr3_nt: String,
    #[pyo3(get, set)]
    pub cdr3_aa: Option<String>,
    #[pyo3(get, set)]
    pub full_seq: String,
    #[pyo3(get, set)]
    pub v_gene: String,
    #[pyo3(get, set)]
    pub j_gene: String,
}

#[pyclass]
pub struct GeneratorVDJ {
    model: ModelVDJ,
    rng: SmallRng,
}

#[pyclass]
pub struct GeneratorVJ {
    model: ModelVJ,
    rng: SmallRng,
}

#[pymethods]
impl GeneratorVDJ {
    #[new]
    pub fn new(
        path_params: &str,
        path_marginals: &str,
        path_v_anchors: &str,
        path_j_anchors: &str,
        seed: Option<u64>,
    ) -> GeneratorVDJ {
        let model = ModelVDJ::load_model(
            Path::new(path_params),
            Path::new(path_marginals),
            Path::new(path_v_anchors),
            Path::new(path_j_anchors),
        )
        .unwrap();
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };

        GeneratorVDJ { model, rng }
    }

    pub fn generate(&mut self, functional: bool) -> GenerationResult {
        let (cdr3_nt, cdr3_aa, v_index, j_index) = self.model.generate(functional, &mut self.rng);
        let (full_sequence, v_name, j_name) = self
            .model
            .recreate_full_sequence(&cdr3_nt, v_index, j_index);
        GenerationResult {
            full_seq: full_sequence.to_string(),
            cdr3_nt: cdr3_nt.to_string(),
            cdr3_aa: cdr3_aa.map(|x: AminoAcid| x.to_string()),
            v_gene: v_name,
            j_gene: j_name,
        }
    }
}

#[pymethods]
impl GeneratorVJ {
    #[new]
    fn new(
        path_params: &str,
        path_marginals: &str,
        path_v_anchors: &str,
        path_j_anchors: &str,
        seed: Option<u64>,
    ) -> GeneratorVJ {
        let model = ModelVJ::load_model(
            Path::new(path_params),
            Path::new(path_marginals),
            Path::new(path_v_anchors),
            Path::new(path_j_anchors),
        )
        .unwrap();
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };
        GeneratorVJ { model, rng }
    }

    fn generate(&mut self, functional: bool) -> GenerationResult {
        let (cdr3_nt, cdr3_aa, v_index, j_index) = self.model.generate(functional, &mut self.rng);
        let (full_sequence, v_name, j_name) = self
            .model
            .recreate_full_sequence(&cdr3_nt, v_index, j_index);
        GenerationResult {
            full_seq: full_sequence.to_string(),
            cdr3_nt: cdr3_nt.to_string(),
            cdr3_aa: cdr3_aa.map(|x| x.to_string()),
            v_gene: v_name,
            j_gene: j_name,
        }
    }
}

// getter & setter for the numpy/ndarray arrays, no easy way to make them automatically
#[pymethods]
impl ModelVJ {
    #[getter]
    fn get_p_v(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_v.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_p_v(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_v = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_p_j_given_v(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_j_given_v.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_p_j_given_v(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_j_given_v = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_p_ins_vj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_ins_vj.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_p_ins_vj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_ins_vj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_p_del_v_given_v(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_del_v_given_v.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_p_del_v_given_v(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_del_v_given_v = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_p_del_j_given_j(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_del_j_given_j.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_p_del_j_given_j(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_del_j_given_j = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_markov_coefficients_vj(&self, py: Python) -> Py<PyArray2<f64>> {
        self.markov_coefficients_vj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[setter]
    fn set_markov_coefficients_vj(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.markov_coefficients_vj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_first_nt_bias_ins_vj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.first_nt_bias_ins_vj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[setter]
    fn set_first_nt_bias_ins_vj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.first_nt_bias_ins_vj = value.as_ref(py).to_owned_array();
        Ok(())
    }
}

#[pymethods]
impl ModelVDJ {
    pub fn update(&mut self, feature: &FeaturesVDJ) {
        self.p_v = feature.v.probas.clone();
        self.p_del_v_given_v = feature.delv.probas.clone();
        self.p_dj = feature.dj.probas.clone();
        self.p_del_j_given_j = feature.delj.probas.clone();
        self.p_del_d3_del_d5 = feature.deld.probas.clone();
        self.p_ins_vd = feature.nb_insvd.probas.clone();
        self.p_ins_dj = feature.nb_insdj.probas.clone();
        (self.first_nt_bias_ins_vd, self.markov_coefficients_vd) = feature.insvd.get_parameters();
        (self.first_nt_bias_ins_dj, self.markov_coefficients_dj) = feature.insvd.get_parameters();
        self.error_rate = feature.error.error_rate;
    }

    #[staticmethod]
    #[pyo3(name = "load_model")]
    pub fn py_load_model(
        path_params: &str,
        path_marginals: &str,
        path_anchor_vgene: &str,
        path_anchor_jgene: &str,
    ) -> Result<ModelVDJ> {
        ModelVDJ::load_model(
            Path::new(path_params),
            Path::new(path_marginals),
            Path::new(path_anchor_vgene),
            Path::new(path_anchor_jgene),
        )
    }

    #[getter]
    fn get_p_v(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_v.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_p_v(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_v = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_p_dj(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_dj.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_p_dj(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_dj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_p_ins_vd(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_ins_vd.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_p_ins_vd(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_ins_vd = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_p_ins_dj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_ins_dj.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_p_ins_dj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_ins_dj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_p_del_v_given_v(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_del_v_given_v.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_p_del_v_given_v(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_del_v_given_v = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_p_del_j_given_j(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_del_j_given_j.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_p_del_j_given_j(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_del_j_given_j = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_p_del_d3_del_d5(&self, py: Python) -> Py<PyArray3<f64>> {
        self.p_del_d3_del_d5.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_p_del_d3_del_d5(&mut self, py: Python, value: Py<PyArray3<f64>>) -> PyResult<()> {
        self.p_del_d3_del_d5 = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_markov_coefficients_vd(&self, py: Python) -> Py<PyArray2<f64>> {
        self.markov_coefficients_vd
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[setter]
    fn set_markov_coefficients_vd(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.markov_coefficients_vd = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_markov_coefficients_dj(&self, py: Python) -> Py<PyArray2<f64>> {
        self.markov_coefficients_dj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[setter]
    fn set_markov_coefficients_dj(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.markov_coefficients_dj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_first_nt_bias_ins_vd(&self, py: Python) -> Py<PyArray1<f64>> {
        self.first_nt_bias_ins_vd
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[setter]
    fn set_first_nt_bias_ins_vd(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.first_nt_bias_ins_vd = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[getter]
    fn get_first_nt_bias_ins_dj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.first_nt_bias_ins_dj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[setter]
    fn set_first_nt_bias_ins_dj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.first_nt_bias_ins_dj = value.as_ref(py).to_owned_array();
        Ok(())
    }
}

#[pymethods]
impl CategoricalFeature1 {
    #[getter]
    fn get_probas(&self, py: Python) -> Py<PyArray1<f64>> {
        self.probas.to_owned().into_pyarray(py).to_owned()
    }
}

#[pymethods]
impl CategoricalFeature1g1 {
    #[getter]
    fn get_probas(&self, py: Python) -> Py<PyArray2<f64>> {
        self.probas.to_owned().into_pyarray(py).to_owned()
    }
}

#[pymethods]
impl CategoricalFeature2 {
    #[getter]
    fn get_probas(&self, py: Python) -> Py<PyArray2<f64>> {
        self.probas.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_probas(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.probas = value.as_ref(py).to_owned_array();
        Ok(())
    }
}

#[pymethods]
impl CategoricalFeature2g1 {
    #[getter]
    fn get_probas(&self, py: Python) -> Py<PyArray3<f64>> {
        self.probas.to_owned().into_pyarray(py).to_owned()
    }
}

#[pymethods]
impl MarkovFeature {
    #[getter]
    fn get_initial_distribution(&self, py: Python) -> Py<PyArray1<f64>> {
        self.initial_distribution
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[getter]
    fn get_transition_matrix(&self, py: Python) -> Py<PyArray2<f64>> {
        self.transition_matrix
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
}
