#![warn(clippy::all)]

pub mod shared;
pub mod v_dj;
pub mod vdj;
pub mod vj;

use anyhow::Result;
use righor::shared::genes_matching;
use righor::shared::Model;
use righor::AminoAcid;
use righor::EntrySequence;
use righor::{AlignmentParameters, InferenceParameters};

use std::path::Path;

fn main() -> Result<()> {
    let mut model = Model::load_from_name(
        "human",
        "trb",
        None,
        &Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/righor.data")),
    )?;

    Ok(())
}
