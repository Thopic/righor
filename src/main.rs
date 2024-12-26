#![warn(clippy::all)]

pub mod shared;
pub mod v_dj;
pub mod vdj;
pub mod vj;

use anyhow::Result;
use righor::shared::Model;
use righor::shared::{AlignmentParameters, InferenceParameters};
use righor::vdj::model::EntrySequence;

use rand::rngs::StdRng;
use rand::SeedableRng;

use std::path::Path;

fn main() -> Result<()> {
    let mut model = Model::load_from_name(
        "human",
        "trb",
        None,
        &Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/righor.data/data/righor_models"
        )),
    )?;
    model.set_model_type(righor::shared::ModelStructure::VxDJ)?;
    model.set_error(righor::shared::errors::ErrorConstantRate::new(0.).into())?;
    let mut rng = StdRng::seed_from_u64(42);
    let mut ip = InferenceParameters::default();
    ip.pgen_only();
    for _ in 0..100 {
        let res = model.generate(true, &mut rng)?;
        let aa = res.junction_aa.unwrap();
        let es = EntrySequence::NucleotideCDR3((
            righor::AminoAcid::from_string(&aa.clone())?.into(),
            righor::genes_matching(&res.v_gene, &model, false)?,
            righor::genes_matching(&res.j_gene, &model, false)?,
        ));

        model.evaluate(es.clone(), &AlignmentParameters::default(), &ip)?;
    }
    Ok(())
}
