use anyhow::Result;
use righor::shared::{AlignmentParameters, DnaLike, InferenceParameters};
mod common;
use righor::shared::errors::ErrorConstantRate;
use righor::shared::ErrorParameters;
use righor::shared::ModelStructure;
use righor::EntrySequence;
use righor::Modelable;

#[test]
fn evaluate_simple_model_vdj() -> Result<()> {
    let mut model = common::simple_model_vdj();
    model.model_type = ModelStructure::VDJ;
    model.uniform()?;
    let mut model2 = common::simple_model_vdj();
    model2.model_type = ModelStructure::VxDJ;
    model2.uniform()?;

    model.error = ErrorParameters::ConstantRate(ErrorConstantRate::new(0.2));
    model2.error = ErrorParameters::ConstantRate(ErrorConstantRate::new(0.2));

    let mut generator = righor::vdj::Generator::new(model.clone(), Some(42), None, None)?;
    let mut ifp = InferenceParameters::default();

    ifp.min_likelihood = 0.;
    ifp.min_ratio_likelihood = 0.;

    let alp = AlignmentParameters::default();
    for _ in 0..100 {
        let s = righor::Dna::from_string(&generator.generate(false)?.full_seq)?;
        let als = EntrySequence::Aligned(model.align_sequence(DnaLike::from_dna(s.clone()), &alp)?);
        let result_model_vdj = model.evaluate(als.clone(), &alp, &ifp)?;
        let result_model_v_dj = model2.evaluate(als.clone(), &alp, &ifp)?;
        let result_model_brute_force = model.evaluate_brute_force(als.clone(), &alp, &ifp)?;

        println!("VDJ\t{:?}", result_model_vdj.best_event);
        println!("VxDJ\t {:?}", result_model_v_dj.best_event);
        println!("BRUTE\t{:?}", result_model_brute_force.best_event);
        println!("");

        // assert!(
        //     result_model_vdj.best_event.clone().unwrap().end_v
        //         == result_model_brute_force.best_event.clone().unwrap().end_v
        // );

        // assert!(
        //     result_model_vdj.best_event.clone().unwrap().d_index
        //         == result_model_brute_force.best_event.clone().unwrap().d_index
        // );

        // assert!(
        //     result_model_vdj.best_event.clone().unwrap().start_j
        //         == result_model_brute_force.best_event.clone().unwrap().start_j
        // );

        // assert!(
        //     result_model_vdj.best_event.clone().unwrap().start_d
        //         == result_model_brute_force.best_event.clone().unwrap().start_d
        // );

        // assert!(
        //     result_model_vdj.best_event.clone().unwrap().end_d
        //         == result_model_brute_force.best_event.clone().unwrap().end_d
        // );

        assert!((result_model_vdj.likelihood - result_model_v_dj.likelihood).abs() < 1e-12);
        assert!((result_model_vdj.likelihood - result_model_brute_force.likelihood).abs() < 1e-12)
    }
    Ok(())
}
