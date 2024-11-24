//! Variation of the VDJ model for TCR beta chain and IGH heavy chain, where P(V,D,J) = P(V, J) * P(D | J)

pub mod feature;
pub mod inference;

// Re-exporting for public API
use self::feature::AggregatedFeatureStartDAndJ;
pub use self::inference::Features;
