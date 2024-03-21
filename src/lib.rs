pub mod sequence;
pub mod shared;
pub mod vdj;
pub mod vj;

pub use crate::sequence::{AlignmentParameters, AminoAcid, DAlignment, Dna, VJAlignment};
pub use crate::shared::{
    genes_matching, CategoricalFeature1, CategoricalFeature1g1, CategoricalFeature2,
    CategoricalFeature2g1, ErrorSingleNucleotide, Gene, InferenceParameters, InsertionFeature,
};
//#[cfg(all(feature = "py_binds", feature = "pyo3"))]
//use crate::vdj::GenerationResult;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymodule]
fn righor(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let vdj_submod = PyModule::new(py, "vdj")?;
    let vj_submod = PyModule::new(py, "vj")?;

    //    vdj_submod.add_class::<vdj::Generator>()?;
    //    vj_submod.add_class::<vj::Generator>()?;
    // vdj_submod.add_class::<vdj::Sequence>()?;
    //    vj_submod.add_class::<vj::Sequence>()?;
    //    vdj_submod.add_class::<vdj::Features>()?;
    //    vj_submod.add_class::<vj::Features>()?;
    //    vdj_submod.add_class::<vdj::Model>()?;
    //    vj_submod.add_class::<vj::Model>()?;
    m.add_class::<Gene>()?;
    m.add_class::<Dna>()?;
    m.add_class::<AminoAcid>()?;
    m.add_class::<VJAlignment>()?;
    m.add_class::<DAlignment>()?;
    // m.add_class::<InferenceParameters>()?;
    // m.add_class::<AlignmentParameters>()?;
    m.add_class::<CategoricalFeature1>()?;
    m.add_class::<CategoricalFeature1g1>()?;
    m.add_class::<CategoricalFeature2>()?;
    m.add_class::<CategoricalFeature2g1>()?;
    m.add_class::<InsertionFeature>()?;
    m.add_class::<ErrorSingleNucleotide>()?;

    //    m.add_class::<GenerationResult>()?;

    m.add_submodule(vdj_submod)?;
    m.add_submodule(vj_submod)?;

    Ok(())
}
