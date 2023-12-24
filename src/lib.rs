pub mod feature;
pub mod inference;
pub mod model;
pub mod parser;
pub mod py_binding;
pub mod sequence;
pub mod utils;
pub mod utils_sequences;

use feature::{
    CategoricalFeature1, CategoricalFeature1g1, CategoricalFeature2, CategoricalFeature2g1,
    ErrorPoisson, MarkovFeature,
};
use inference::{infer_features, FeaturesVDJ, InferenceParameters};
use model::{ModelVDJ, ModelVJ};
use py_binding::{GenerationResult, GeneratorVDJ, GeneratorVJ};
use pyo3::prelude::*;
use sequence::{DAlignment, SequenceVDJ, VJAlignment};
use utils::Gene;
use utils_sequences::{AlignmentParameters, AminoAcid, Dna};

#[pymodule]
fn ihor(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<GeneratorVDJ>()?;
    m.add_class::<GeneratorVJ>()?;
    m.add_class::<GenerationResult>()?;
    m.add_class::<SequenceVDJ>()?;
    m.add_class::<Gene>()?;
    m.add_class::<Dna>()?;
    m.add_class::<AminoAcid>()?;
    m.add_class::<ModelVDJ>()?;
    m.add_class::<ModelVJ>()?;
    m.add_class::<VJAlignment>()?;
    m.add_class::<DAlignment>()?;
    m.add_class::<InferenceParameters>()?;
    m.add_class::<AlignmentParameters>()?;
    m.add_class::<CategoricalFeature1>()?;
    m.add_class::<CategoricalFeature1g1>()?;
    m.add_class::<CategoricalFeature2>()?;
    m.add_class::<CategoricalFeature2g1>()?;
    m.add_class::<MarkovFeature>()?;
    m.add_class::<ErrorPoisson>()?;
    m.add_class::<FeaturesVDJ>()?;
    m.add_function(wrap_pyfunction!(infer_features, m)?)?;
    Ok(())
}
