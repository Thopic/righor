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

fn main() -> Result<()> {
    let mut model = Model::load_from_name(
        "human",
        "trb",
        None,
        &Path(concat!(env!("CARGO_MANIFEST_DIR"), "/righor.data")),
    )?;

    let ip = InferenceParameters::default();
    let ap = AlignmentParameters::default();

    let v = genes_matching("TRBV13", &model, false)?;
    let j = genes_matching("TRBJ1-2", &model, false)?;
    let seq =
        EntrySequence::NucleotideCDR3((AminoAcid::from_string("CASSYRGQHKSGYTF").into(), v, j));
    let res = model.evaluate(seq, &ap, &ip);
    println!("{:?}", res);
    Ok(())
}
