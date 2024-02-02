#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use crate::shared::feature::{
    CategoricalFeature1, CategoricalFeature1g1, CategoricalFeature2, CategoricalFeature2g1,
    CategoricalFeature3, InsertionFeature,
};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use crate::shared::utils::calc_steady_state_dist;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl CategoricalFeature1 {
    #[getter]
    fn get_probas(&self, py: Python) -> Py<PyArray1<f64>> {
        self.probas.to_owned().into_pyarray(py).to_owned()
    }
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl CategoricalFeature1g1 {
    #[getter]
    fn get_probas(&self, py: Python) -> Py<PyArray2<f64>> {
        self.probas.to_owned().into_pyarray(py).to_owned()
    }
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
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

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl CategoricalFeature3 {
    #[getter]
    fn get_probas(&self, py: Python) -> Py<PyArray3<f64>> {
        self.probas.to_owned().into_pyarray(py).to_owned()
    }
    #[setter]
    fn set_probas(&mut self, py: Python, value: Py<PyArray3<f64>>) -> PyResult<()> {
        self.probas = value.as_ref(py).to_owned_array();
        Ok(())
    }
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl CategoricalFeature2g1 {
    #[getter]
    fn get_probas(&self, py: Python) -> Py<PyArray3<f64>> {
        self.probas.to_owned().into_pyarray(py).to_owned()
    }
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl InsertionFeature {
    #[getter]
    fn get_transition_matrix(&self, py: Python) -> Py<PyArray2<f64>> {
        self.transition_matrix
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[getter]
    fn get_length_distribution(&self, py: Python) -> Py<PyArray1<f64>> {
        self.length_distribution
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[getter]
    fn get_initial_distribution(&self, py: Python) -> Py<PyArray1<f64>> {
        calc_steady_state_dist(&self.transition_matrix)
            .unwrap()
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
}
