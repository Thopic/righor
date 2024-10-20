//! Shared functionalities between VDJ and VJ (not related to alignment)
pub mod alignment;
pub mod data_structures;
pub mod distributions;
pub mod errors;
pub mod event;
pub mod feature;
pub mod gene;
pub mod likelihood;
pub mod markov_chain;
pub mod model;
pub mod parameters;
pub mod parser;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
pub mod py_binding;
pub mod sequence;
pub mod utils;

pub use errors::{ErrorParameters, FeatureError};

pub use event::StaticEvent;
pub use feature::{
    CategoricalFeature1, CategoricalFeature1g1, CategoricalFeature1g2, CategoricalFeature2,
    CategoricalFeature2g1, Feature, Features, InfEvent, InsertionFeature, ResultInference,
};

pub use alignment::{
    DAlignment, ErrorAlignment, ErrorDAlignment, ErrorJAlignment, ErrorVAlignment, VJAlignment,
};
pub use gene::{genes_matching, Gene, ModelGen};
pub use likelihood::{
    Likelihood, Likelihood1DContainer, Likelihood2DContainer, LikelihoodInsContainer,
    LikelihoodType,
};
pub use markov_chain::DNAMarkovChain;
pub use model::{GenerationResult, Generator, Model, ModelStructure, Modelable};
pub use parameters::{AlignmentParameters, InferenceParameters};
pub use sequence::{nucleotides_inv, AminoAcid, Dna, DnaLike};
pub use utils::RecordModel;
