//! VJ model for TCR alpha chain and IGH light chain

pub mod event;
//pub mod inference;
pub mod model;
//pub mod py_bindings;
//pub mod sequence;

// Re-exporting for public API
pub use self::event::StaticEvent;
//pub use self::inference::Features;
pub use self::model::{Generator, Model};
