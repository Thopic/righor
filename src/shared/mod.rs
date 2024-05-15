//! Shared functionalities between VDJ and VJ (not related to alignment)
pub mod data_structures;
pub mod distributions;
pub mod errors;
pub mod event;
pub mod feature;
pub mod gene;
pub mod model;
pub mod parameters;
pub mod parser;
pub mod py_binding;
pub mod sequence;
pub mod utils;

pub use errors::{ErrorParameters, FeatureError};

pub use event::StaticEvent;
pub use feature::{
    CategoricalFeature1, CategoricalFeature1g1, CategoricalFeature1g2, CategoricalFeature2,
    CategoricalFeature2g1, Feature, Features, InfEvent, InsertionFeature, ResultInference,
};

pub use gene::{genes_matching, Gene, ModelGen};
pub use model::{GenerationResult, Model, ModelStructure, Modelable};
pub use parameters::{AlignmentParameters, InferenceParameters};
pub use sequence::{nucleotides_inv, AminoAcid, DAlignment, Dna, ErrorAlignment, VJAlignment};
pub use utils::RecordModel;
