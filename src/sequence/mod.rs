//! All the functionalities to align V(D)J sequence
pub mod sequence;
pub mod utils;

pub use sequence::{DAlignment, VJAlignment};
pub use utils::{AlignmentParameters, AminoAcid, Dna};
