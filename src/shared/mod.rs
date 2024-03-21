//! Shared functionalities between VDJ and VJ (not related to alignment)
pub mod feature;
pub mod model;
pub mod parser;
pub mod py_binding;
pub mod utils;

pub use feature::{
    CategoricalFeature1, CategoricalFeature1g1, CategoricalFeature1g2, CategoricalFeature2,
    CategoricalFeature2g1, ErrorSingleNucleotide, InsertionFeature,
};
pub use utils::{genes_matching, Gene, InferenceParameters};
