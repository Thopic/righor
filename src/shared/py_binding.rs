#[cfg(all(feature = "py_binds", feature = "py_o3"))]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
#[cfg(all(feature = "py_binds", feature = "py_o3"))]
use pyo3::prelude::*;

#[cfg_attr(
    all(feature = "py_binds", feature = "py_o3"),
    pyclass(get_all, set_all)
)]
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub cdr3_nt: String,
    pub cdr3_aa: Option<String>,
    pub full_seq: String,
    pub v_gene: String,
    pub j_gene: String,
}

#[cfg(all(feature = "py_binds", feature = "py_o3"))]
#[pymethods]
impl CategoricalFeature1 {
    #[getter]
    fn get_probas(&self, py: Python) -> Py<PyArray1<f64>> {
        self.probas.to_owned().into_pyarray(py).to_owned()
    }
}

#[cfg(all(feature = "py_binds", feature = "py_o3"))]
#[pymethods]
impl CategoricalFeature1g1 {
    #[getter]
    fn get_probas(&self, py: Python) -> Py<PyArray2<f64>> {
        self.probas.to_owned().into_pyarray(py).to_owned()
    }
}

#[cfg(all(feature = "py_binds", feature = "py_o3"))]
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

#[cfg(all(feature = "py_binds", feature = "py_o3"))]
#[pymethods]
impl CategoricalFeature2g1 {
    #[getter]
    fn get_probas(&self, py: Python) -> Py<PyArray3<f64>> {
        self.probas.to_owned().into_pyarray(py).to_owned()
    }
}

#[cfg(all(feature = "py_binds", feature = "py_o3"))]
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
