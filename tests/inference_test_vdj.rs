use anyhow::Result;
use righor::shared::errors::ErrorConstantRate;
use righor::shared::DnaLike;
use righor::shared::ErrorParameters;
use righor::shared::ModelStructure;
use righor::shared::{AlignmentParameters, InferenceParameters};
use righor::EntrySequence;
use righor::Modelable;

mod common;

#[test]
fn infer_comparison_v_dj_vdj_simple_model() -> Result<()> {
    let mut model = common::simple_model_vdj();
    let mut model2 = model.clone();
    let mut generator = righor::vdj::Generator::new(model.clone(), Some(48), None, None)?;
    let ifp = InferenceParameters::default();
    let ifp_2 = InferenceParameters::default();
    let alp = AlignmentParameters::default();
    let mut alignments = Vec::new();
    for _ in 0..10 {
        let s = righor::Dna::from_string(&generator.generate(false)?.full_seq)?;
        let als = model.align_sequence(DnaLike::from_dna(s.clone()), &alp)?;
        alignments.push(EntrySequence::Aligned(als));
    }

    model.infer(&alignments.clone(), None, &alp, &ifp)?;
    model2.infer(&alignments.clone(), None, &alp, &ifp_2)?;

    assert!(model.p_ins_vd.abs_diff_eq(&model2.p_ins_vd, 1e-12));
    Ok(())
}

#[test]
fn infer_vs_brute_force() -> Result<()> {
    let mut model = common::simple_model_vdj();
    model.model_type = ModelStructure::VDJ;
    model.error = ErrorParameters::ConstantRate(ErrorConstantRate::new(0.1));
    let uniform_model = model.uniform()?.clone();
    let mut generator = righor::vdj::Generator::new(model.clone(), Some(42), None, None)?;
    let ifp = InferenceParameters::default();

    let alp = AlignmentParameters::default();
    let mut alignments = Vec::new();
    for _ in 0..20 {
        let generated = generator.generate(false)?;
        let s = righor::Dna::from_string(&generated.full_seq)?;
        let als = EntrySequence::Aligned(model.align_sequence(DnaLike::from_dna(s.clone()), &alp)?);
        alignments.push(als.clone());

        let a1 = uniform_model.evaluate(als.clone(), &alp, &ifp)?.likelihood;
        let a2 = uniform_model
            .evaluate_brute_force(als.clone(), &alp, &ifp)?
            .likelihood;

        assert!((a1 - a2).abs() < 1e-12);
    }

    let mut inferred_model = model.uniform()?;
    //    inferred_model.error_rate = 0.;
    for _ in 0..1 {
        inferred_model.infer_brute_force(&alignments.clone(), None, &alp, &ifp)?;
    }
    let mut inferred_model2 = model.uniform()?;
    //    inferred_model2.error_rate = 0.;
    for _ in 0..1 {
        inferred_model2.infer(&alignments.clone(), None, &alp, &ifp)?;
    }

    println!("{:?}", inferred_model.p_ins_vd);
    println!("{:?}", inferred_model2.p_ins_vd);
    println!();
    println!("{:?}", inferred_model.p_ins_dj);
    println!("{:?}", inferred_model2.p_ins_dj);
    println!();

    println!("{:?}", inferred_model.p_del_v_given_v);
    println!("{:?}", inferred_model2.p_del_v_given_v);
    println!();
    println!("{:?}", inferred_model.p_del_j_given_j);
    println!("{:?}", inferred_model2.p_del_j_given_j);
    println!();
    println!("{:?}", inferred_model.p_vdj);
    println!("{:?}", inferred_model2.p_vdj);

    println!();
    println!("{:?}", inferred_model.p_del_d5_del_d3);
    println!("{:?}", inferred_model2.p_del_d5_del_d3);

    println!();
    println!("{:?}", inferred_model.error);
    println!("{:?}", inferred_model2.error);

    assert!(inferred_model2
        .p_ins_vd
        .abs_diff_eq(&inferred_model.p_ins_vd, 1e-12));

    println!("{}", inferred_model2.similar_to(inferred_model));
    Ok(())
}

#[test]
fn full_inference_simple_model() -> Result<()> {
    let mut model = common::simple_model_vdj();
    model.model_type = ModelStructure::VDJ;
    model.error = ErrorParameters::ConstantRate(ErrorConstantRate::new(0.1));
    let mut generator = righor::vdj::Generator::new(model.clone(), Some(48), None, None)?;
    let ifp = InferenceParameters::default();
    let alp = AlignmentParameters::default();
    let mut alignments = Vec::new();
    for _ in 0..1000 {
        let generated = generator.generate(false)?;
        let s = righor::Dna::from_string(&generated.full_seq)?;
        let als = model.align_sequence(DnaLike::from_dna(s.clone()), &alp)?;
        alignments.push(EntrySequence::Aligned(als));
    }

    println!("INFERENCE");
    let mut inferred_model = model.uniform()?;
    inferred_model.error = ErrorParameters::ConstantRate(ErrorConstantRate::new(0.3));
    for _ in 0..10 {
        inferred_model.infer(&alignments.clone(), None, &alp, &ifp)?;
    }

    println!("{:?}", inferred_model.error);
    println!("{:?}", model.error);

    Ok(())
}
