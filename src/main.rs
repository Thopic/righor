#![warn(clippy::all)]

pub mod shared;
pub mod v_dj;
pub mod vdj;
pub mod vj;

use anyhow::Result;
use ndarray::Axis;
use righor::shared::GenerationResult;
use righor::vdj::model::EntrySequence;
use righor::vdj::Generator;
use righor::{
    shared::{errors::ErrorConstantRate, model::simple_model},
    Modelable,
};

use righor::Dna;
use righor::{AlignmentParameters, InferenceParameters};

fn main() -> Result<()> {
    let mut model = simple_model();
    let model_freeze = model.clone();
    model.error = ErrorConstantRate::new(0.).into();

    let mut gen = Generator::new(&model, Some(42), None, None)?;
    let vec: Vec<GenerationResult> = (0..10000)
        .map(|_| gen.generate(false))
        .collect::<Result<Vec<_>>>()?;

    let sequences: Vec<EntrySequence> = vec
        .iter()
        .map(|x| EntrySequence::NucleotideSequence(Dna::from_string(&x.full_seq).unwrap().into()))
        .collect();

    let ap = AlignmentParameters::default();
    let ip = InferenceParameters::default();
    println!(
        "{:.2e}",
        model.clone().p_del_d5_del_d3 - model_freeze.clone().p_del_d5_del_d3
    );

    for _ in 0..3 {
        let (_feat, _) = model.infer(&sequences, None, &ap, &ip)?;

        println!(
            "{:.2e}",
            model.p_del_d5_del_d3.clone() // - model_freeze.clone().p_del_d5_del_d3
        );

        println!(
            "{:.2e}",
            (model.p_vdj.clone() - model_freeze.clone().p_vdj)
                .sum_axis(Axis(0))
                .sum_axis(Axis(1))
        );
    }
    //println!("{:.2e}", model_freeze.p_del_d5_del_d3);

    Ok(())
}
