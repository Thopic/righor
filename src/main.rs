#![warn(clippy::all)]

pub mod shared;
pub mod v_dj;
pub mod vdj;
pub mod vj;

use anyhow::Result;
use kdam::tqdm;
use shared::errors::ErrorConstantRate;
use std::path::Path;

// generate 1'000 nucleotide sequences and infer them
fn run_infer() -> Result<()> {
    let mut model = shared::Model::load_from_name(
        "human",
        "igh",
        None,
        &Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/righor.data/data/righor_models/"
        )),
    )?;
    let mut generator = shared::Generator::new(&model.clone(), Some(48), None, None)?;
    //let mut als = Vec::new();
    let mut als = Vec::new();
    let ifp = shared::InferenceParameters::default();
    let alp = shared::AlignmentParameters::default();
    for _ in tqdm!(0..1000) {
        let generated = generator.generate(false)?;
        let es = shared::EntrySequence::NucleotideCDR3((
            shared::Dna::from_string(&generated.junction_nt)?.into(),
            model
                .get_v_segments()
                .into_iter()
                .filter(|x| x.name == generated.v_gene)
                .collect(),
            model
                .get_j_segments()
                .into_iter()
                .filter(|x| x.name == generated.j_gene)
                .collect(),
        ));
        als.push(es)
    }
    model = model.uniform()?;
    model.infer(&als, None, &alp, &ifp)?;
    Ok(())
}

// generate 10'000 nucleotide sequences and infer them
fn run_evaluate() -> Result<()> {
    let mut model = shared::Model::load_from_name(
        "human",
        "igh",
        None,
        &Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/righor.data/data/righor_models/"
        )),
    )?;
    model.set_error(shared::ErrorParameters::ConstantRate(
        ErrorConstantRate::new(0.),
    ));
    //model.set_model_type(shared::ModelStructure::VDJ);

    let mut generator = shared::Generator::new(&model.clone(), Some(48), None, None)?;
    let mut ifp = shared::InferenceParameters::default();
    ifp.store_best_event = false;
    ifp.do_not_infer_features();
    let alp = shared::AlignmentParameters::default();
    for _ in tqdm!(0..10000) {
        let generated = generator.generate(false)?;
        let es = shared::EntrySequence::NucleotideCDR3((
            shared::Dna::from_string(&generated.junction_nt)?.into(),
            model
                .get_v_segments()
                .into_iter()
                .filter(|x| x.name == generated.v_gene)
                .collect(),
            model
                .get_j_segments()
                .into_iter()
                .filter(|x| x.name == generated.j_gene)
                .collect(),
        ));
        model.evaluate(es, &alp, &ifp);
    }
    Ok(())
}

// generate 1'000 nucleotide sequences and evaluate them
fn run_evaluate_aa() -> Result<()> {
    let mut model = shared::Model::load_from_name(
        "human",
        "trb",
        None,
        &Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/righor.data/data/righor_models/"
        )),
    )?;

    model.set_error(shared::ErrorParameters::ConstantRate(
        ErrorConstantRate::new(0.),
    ))?;

    let mut generator = shared::Generator::new(&model.clone(), Some(48), None, None)?;
    let mut ifp = shared::InferenceParameters::default();
    ifp.store_best_event = false;
    ifp.min_ratio_likelihood = 1e-2;
    let alp = shared::AlignmentParameters::default();
    for _ in tqdm!(0..1000) {
        let generated = generator.generate(true)?;
        let es = shared::EntrySequence::NucleotideCDR3((
            shared::AminoAcid::from_string(&generated.junction_aa.unwrap())?.into(),
            model
                .get_v_segments()
                .into_iter()
                .filter(|x| x.name == generated.v_gene)
                .collect(),
            model
                .get_j_segments()
                .into_iter()
                .filter(|x| x.name == generated.j_gene)
                .collect(),
        ));

        let r = model.evaluate(es, &alp, &ifp)?;
        //println!("{:?}", r);
    }

    Ok(())
}

fn main() -> Result<()> {
    let _ = run_evaluate()?;
    Ok(())
}
