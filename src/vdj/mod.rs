//! VDJ model for TCR beta chain and IGH heavy chain

pub mod event;
pub mod feature;
pub mod inference;
pub mod model;
pub mod sequence;

// Re-exporting for public API
pub use self::event::{Event, StaticEvent};
use self::feature::{
    AggregatedFeatureEndV, AggregatedFeatureSpanD, AggregatedFeatureStartJ, FeatureDJ, FeatureVD,
};
pub use self::inference::{Features, InfEvent, ResultHuman, ResultInference};
pub use self::model::{GenerationResult, Generative, Generator, Model};
pub use self::sequence::{display_j_alignment, display_v_alignment, Sequence};
