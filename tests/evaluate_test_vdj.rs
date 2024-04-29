use anyhow::Result;
use righor::shared::{AlignmentParameters, InferenceParameters};
mod common;
use righor::Modelable;

#[test]
fn evaluate_simple_model_vdj() -> Result<()> {
    let mut model = common::simple_model_vdj();
    model.uniform()?;
    model.error_rate = 0.2;
    let mut generator = righor::vdj::Generator::new(model.clone(), Some(48), None, None)?;
    let ifp = InferenceParameters::default();
    let ifp_2 = InferenceParameters {
        complete_vdj_inference: true,
        ..Default::default()
    };
    let alp = AlignmentParameters::default();
    for _ in 0..100 {
        let s = righor::Dna::from_string(&generator.generate(false).full_seq)?;
        let als = model.align_sequence(&s.clone(), &alp)?;

        let likelihood_model_vdj = model.evaluate(&als.clone(), &ifp_2)?.likelihood;
        let likelihood_model_v_dj = model.evaluate(&als.clone(), &ifp)?.likelihood;
        let likelihood_model_brute_force = model.evaluate_brute_force(&als.clone())?.likelihood;

        println!("{}", likelihood_model_vdj);
        println!("{}", likelihood_model_v_dj);
        println!("{}", likelihood_model_brute_force);

        assert!((likelihood_model_vdj - likelihood_model_v_dj).abs() < 1e-12);
        assert!((likelihood_model_vdj - likelihood_model_brute_force).abs() < 1e-12)
    }
    Ok(())
}
