use crate::shared::feature::{
    CategoricalFeature1, CategoricalFeature1g1, CategoricalFeature2, CategoricalFeature2g1,
    MarkovFeature,
};

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;

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
