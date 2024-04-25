pub mod shared;
pub mod v_dj;
pub mod vdj;
pub mod vj;

pub use crate::shared::{
    genes_matching, AlignmentParameters, AminoAcid, CategoricalFeature1, CategoricalFeature1g1,
    CategoricalFeature2, CategoricalFeature2g1, DAlignment, Dna, ErrorSingleNucleotide, Gene,
    InferenceParameters, InsertionFeature, Modelable, VJAlignment,
};

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymodule]
fn righor(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let vdj_submod = PyModule::new(py, "vdj")?;
    let vj_submod = PyModule::new(py, "vj")?;

    m.add_class::<Gene>()?;
    m.add_class::<Dna>()?;
    m.add_class::<AminoAcid>()?;
    m.add_class::<VJAlignment>()?;
    m.add_class::<DAlignment>()?;
    m.add_class::<CategoricalFeature1>()?;
    m.add_class::<CategoricalFeature1g1>()?;
    m.add_class::<CategoricalFeature2>()?;
    m.add_class::<CategoricalFeature2g1>()?;
    m.add_class::<InsertionFeature>()?;
    m.add_class::<ErrorSingleNucleotide>()?;
    m.add_submodule(vdj_submod)?;
    m.add_submodule(vj_submod)?;

    Ok(())
}
