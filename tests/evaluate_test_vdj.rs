use anyhow::Result;
use righor::shared::likelihood::Vector16;
use righor::shared::sequence::DegenerateCodonSequence;
use righor::shared::sequence::DnaLikeEnum;
use righor::shared::{AlignmentParameters, DnaLike, InferenceParameters};
mod common;
use kdam::tqdm;
use righor::shared::errors::ErrorConstantRate;
use righor::shared::likelihood::Likelihood;

use ndarray::array;
use righor::shared::DNAMarkovChain;
use righor::shared::ErrorParameters;
use righor::shared::ModelStructure;
use righor::EntrySequence;
use righor::Modelable;
use righor::{AminoAcid, Dna};

use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn evaluate_cdr3() -> Result<()> {
    let mut model = common::simple_model_vdj();
    // ATCTACTACTACTGCTCATGCAAATTTTTCGCTTTTGGGGCAGTCTTCGGAGAAACAAAGACTTAT
    // nt:               TGCAAATTTTTCGCTTTTGGGGCAGTCTTC
    //                         TTTTTCGCTTTT
    //                                   GGGGGGCAGTCTTCGGAGAAACAAAGACTTAT
    // ATCTACTACTACTGCTCATGCAAAAAAAAATT
    let dna = Dna::from_string("TGCAAATTTTTCGCTTTTGGGGCAGTCTTC")?;
    let cdr3 = dna.clone().translate()?;
    println!("{}", cdr3.to_string());
    let es = EntrySequence::NucleotideCDR3((
        cdr3.clone().into(),
        model.clone().seg_vs,
        model.clone().seg_js,
    ));
    let mut ip = InferenceParameters::default();
    ip.infer_features = false;

    let result = model.evaluate(es, &AlignmentParameters::default(), &ip);
    println!("{:?}\n", result);

    let es = EntrySequence::NucleotideCDR3((
        dna.clone().into(),
        model.clone().seg_vs,
        model.clone().seg_js,
    ));
    let result = model.evaluate(es, &AlignmentParameters::default(), &ip);
    println!("{:?}\n", result);

    let es = EntrySequence::NucleotideSequence(
        Dna::from_string("ATCTACTACTACTGCTCATGCAAATTTTTCGCTTTTGGGGCAGTCTTCGGAGAAACAAAGACTTAT")?
            .into(),
    );

    let result = model.evaluate(es, &AlignmentParameters::default(), &ip)?;
    println!("{:?}", result);

    println!("{}", cdr3.clone().to_dnas().len());
    let mut sum = 0.;
    for nt in &cdr3.to_dnas() {
        //	println!("{}", nt.to_string());
        let es = EntrySequence::NucleotideCDR3((
            nt.clone().into(),
            model.clone().seg_vs,
            model.clone().seg_js,
        ));
        //println!("SEQ: {}", nt.to_string());
        let result = model.evaluate(es, &AlignmentParameters::default(), &ip)?;
        sum += result.likelihood;
        if result.likelihood > 0. {
            println!("{} {}", nt.to_string(), result.likelihood);
        }
    }
    println!("{}", sum);

    let dna = Dna::from_string("TGCAAATATTTTTCGCTTTTGGCAGTCTTC")?;
    let es = EntrySequence::NucleotideCDR3((
        dna.clone().into(),
        model.clone().seg_vs,
        model.clone().seg_js,
    ));
    let mut ip = InferenceParameters::default();
    ip.infer_features = true;

    model.model_type = righor::shared::ModelStructure::VDJ;
    let result = model.evaluate(es.clone(), &AlignmentParameters::default(), &ip);
    println!("VDJ pgen(TGCGCCAGGAACTATGAACAGTATTTT) {:?}\n", result);
    model.model_type = righor::shared::ModelStructure::VxDJ;
    let result = model.evaluate(es.clone(), &AlignmentParameters::default(), &ip);
    println!("VxDJ pgen(TGCGCCAGGAACTATGAACAGTATTTT) {:?}\n", result);

    Ok(())
}

#[test]
fn evaluate_degenerate_seq() -> Result<()> {
    let ip = InferenceParameters::default();
    let model = common::simple_model_vdj();
    let es = EntrySequence::NucleotideSequence(
        Dna::from_string("ATCTACTACTACTGCTCATGCAANNNTTTCGCTTTTGGGGCAGTCTTCGGAGAAACAAAGACTTAT")?
            .into(),
    );
    let result = model.evaluate(es, &AlignmentParameters::default(), &ip)?;
    println!("{:?}", result);

    let mut total_likelihood = 0.;
    for a1 in ['A', 'C', 'G', 'T'] {
        for a2 in ['A', 'C', 'G', 'T'] {
            for a3 in ['A', 'C', 'G', 'T'] {
                let seq = format!(
                    "ATCTACTACTACTGCTCATGCAA{}{}{}TTTCGCTTTTGGGGCAGTCTTCGGAGAAACAAAGACTTAT",
                    a1, a2, a3
                );
                let result2 = model.evaluate(
                    EntrySequence::NucleotideSequence(Dna::from_string(&seq)?.into()),
                    &AlignmentParameters::default(),
                    &ip,
                )?;
                total_likelihood += result2.likelihood;
            }
        }
    }
    println!("{:.3e}", total_likelihood);
    println!("{:.3e}", result.likelihood);

    if ((total_likelihood - result.likelihood) / (result.likelihood + total_likelihood)).abs()
        > 0.001
    {
        println!("{:?}", result);
        println!("{:.3e}", total_likelihood);
        println!("{:.3e}", result.likelihood);
    }
    assert!(
        ((total_likelihood - result.likelihood) / (total_likelihood + result.likelihood)).abs()
            < 0.001
    );
    Ok(())
}

#[test]
fn evaluate_many_cdr3() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut model = common::simple_model_vdj();
    let ip = InferenceParameters::default();

    for _ in tqdm!(0..1000) {
        let res = model.generate_no_error(false, &mut rng);
        let event = res.3;
        let dna = res.1;
        let es = EntrySequence::NucleotideCDR3((
            dna.clone().into(),
            vec![model.clone().seg_vs[event.v_index].clone()],
            vec![model.clone().seg_js[event.j_index].clone()],
        ));

        model.model_type = righor::shared::ModelStructure::VDJ;
        let result1 = model.evaluate(es.clone(), &AlignmentParameters::default(), &ip)?;
        model.model_type = righor::shared::ModelStructure::VxDJ;
        let result2 = model.evaluate(es.clone(), &AlignmentParameters::default(), &ip)?;

        if ((result1.likelihood - result2.likelihood) / (result1.likelihood + result2.likelihood))
            .abs()
            > 0.001
        {
            println!("CDR3 sequence {:?}", dna);
            println!("V / J {} / {}", event.v_index, event.j_index);
            println!("{:?}", result1);
            println!("{:?}", result2);
        }
        assert!(
            ((result1.likelihood - result2.likelihood) / (result1.likelihood + result2.likelihood))
                .abs()
                < 0.001
        );
    }
    Ok(())
}

#[test]
fn evaluate_many_cdr3_aa() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut model = common::simple_model_vdj();
    let ip = InferenceParameters::default();

    for _ in tqdm!(0..100) {
        let res = model.generate_no_error(true, &mut rng);
        let event = res.3;
        let aa = res.2.unwrap();
        let es = EntrySequence::NucleotideCDR3((
            aa.clone().into(),
            vec![model.clone().seg_vs[event.v_index].clone()],
            vec![model.clone().seg_js[event.j_index].clone()],
        ));

        model.model_type = righor::shared::ModelStructure::VDJ;
        let result1 = model.evaluate(es.clone(), &AlignmentParameters::default(), &ip)?;
        model.model_type = righor::shared::ModelStructure::VxDJ;
        let result2 = model.evaluate(es.clone(), &AlignmentParameters::default(), &ip)?;

        if ((result1.likelihood - result2.likelihood) / (result1.likelihood + result2.likelihood))
            .abs()
            > 0.001
        {
            println!("CDR3 sequence {:?}", aa);
            println!("V / J {} / {}", event.v_index, event.j_index);
            println!("{:?}", result1);
            println!("{:?}", result2);
        }
        assert!(
            ((result1.likelihood - result2.likelihood) / (result1.likelihood + result2.likelihood))
                .abs()
                < 0.001
        );
    }
    Ok(())
}

#[test]
fn evaluate_many_cdr3_vs_aa() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(40);
    let mut model = common::simple_aa_model_vdj()?;
    let ip = InferenceParameters::default();

    for _ in tqdm!(0..100) {
        let mut res = model.generate_no_error(true, &mut rng);
        let mut event = res.clone().3;
        let mut aa = res.clone().2.unwrap();
        while aa.to_string().len() > 5 {
            res = model.generate_no_error(true, &mut rng);
            event = res.clone().3;
            aa = res.clone().2.unwrap();
        }

        model.model_type = righor::shared::ModelStructure::VDJ;
        println!("\n");
        let mut total_likelihood = 0.;

        for dna in aa.clone().to_dnas().iter() {
            let es = EntrySequence::NucleotideCDR3((
                dna.clone().into(),
                vec![model.clone().seg_vs[event.v_index].clone()],
                vec![model.clone().seg_js[event.j_index].clone()],
            ));
            let result = model.evaluate(es.clone(), &AlignmentParameters::default(), &ip)?;
            println!("{:?}", result);
            total_likelihood += result.likelihood;
        }

        let es = EntrySequence::NucleotideCDR3((
            aa.clone().into(),
            vec![model.clone().seg_vs[event.v_index].clone()],
            vec![model.clone().seg_js[event.j_index].clone()],
        ));
        let result2 = model.evaluate(es.clone(), &AlignmentParameters::default(), &ip)?;

        if ((total_likelihood - result2.likelihood) / (result2.likelihood + total_likelihood)).abs()
            > 0.001
        {
            println!("CDR3 sequence {:?}", aa);
            println!("{:?}", res);
            println!("{:?}", result2);
            println!("V / J {} / {}", event.v_index, event.j_index);
            println!("{:.3e}", total_likelihood);
            println!("{:.3e}", result2.likelihood);
        }
        assert!(
            ((total_likelihood - result2.likelihood) / (total_likelihood + result2.likelihood))
                .abs()
                < 0.001
        );
    }
    Ok(())
}

#[test]
fn evaluate_markov_amino_acid() -> Result<()> {
    let epsilon = 1e-10;
    let array = array![
        [0.2, 0.5, 0.3, 0.7],
        [0.3, 0.5, 0.3, 0.7],
        [0.4, 0.1, 0.4, 0.01],
        [0.5, 0.8, 0.4, 0.002]
    ];

    let array = array![
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.3, 0.2],
    ];

    let mkc = DNAMarkovChain::new(&array, false)?;

    let aa = AminoAcid::from_string("WMM")?;
    let seq = DegenerateCodonSequence::from_aminoacid(aa.clone());
    let raa = mkc.likelihood_aminoacid(&seq, 0);
    let mut rnt = Likelihood::Scalar(0.);
    for seqnt in aa.to_dnas() {
        rnt += mkc.likelihood_dna(&seqnt, 0);
    }
    assert!(((raa.to_matrix()? * Vector16::repeat(1.)).max() - rnt.max()).abs() < epsilon);

    let aa = AminoAcid::from_string("CFW")?;
    let seq = DegenerateCodonSequence::from_aminoacid(aa.clone());
    let raa = mkc.likelihood_aminoacid(&seq, 0);
    let mut rnt = Likelihood::Scalar(0.);
    for seqnt in aa.to_dnas() {
        rnt += mkc.likelihood_dna(&seqnt, 0);
    }
    assert!(((raa.to_matrix()? * Vector16::repeat(1.)).max() - rnt.max()).abs() < epsilon);

    let aa = AminoAcid::from_string("WML")?;
    let mut seq = DegenerateCodonSequence::from_aminoacid(aa.clone());
    let raa = mkc.likelihood_aminoacid(&seq, 0);
    let mut rnt = Likelihood::Scalar(0.);
    for seqnt in aa.to_dnas() {
        rnt += mkc.likelihood_dna(&seqnt, 0);
    }
    assert!(((raa.to_matrix()? * Vector16::repeat(1.)).max() - rnt.max()).abs() < epsilon);

    let aa = AminoAcid::from_string("LML")?;
    let mut seq = DegenerateCodonSequence::from_aminoacid(aa.clone());
    let raa = mkc.likelihood_aminoacid(&seq, 0);
    let mut rnt = Likelihood::Scalar(0.);
    for seqnt in aa.to_dnas() {
        rnt += mkc.likelihood_dna(&seqnt, 0);
    }
    assert!(((raa.to_matrix()? * Vector16::repeat(1.)).max() - rnt.max()).abs() < epsilon);

    let aa = AminoAcid::from_string("WML")?;
    let mut seq = DegenerateCodonSequence::from_aminoacid(aa.clone());
    seq = seq.extract_subsequence(1, 9);
    let raa = mkc.likelihood_aminoacid(&seq, 1);
    let mut rnt = Likelihood::Scalar(0.);
    for seqnt in aa.to_dnas() {
        if seqnt.seq[0] == b'C' {
            rnt += mkc.likelihood_dna(&seqnt.extract_subsequence(1, 9), 1);
        }
    }
    assert!(((raa.to_matrix()? * Vector16::repeat(1.)).max() - rnt.max()).abs() < epsilon);

    let aa = AminoAcid::from_string("L")?;
    let seq = DegenerateCodonSequence::from_aminoacid(aa.clone());
    let raa = mkc.likelihood_aminoacid(&seq, 0);
    let mut rnt = Likelihood::Scalar(0.);
    for seqnt in aa.to_dnas() {
        rnt += mkc.likelihood_dna(&seqnt, 0);
    }
    assert!(((raa.to_matrix()? * Vector16::repeat(1.)).max() - rnt.max()).abs() < epsilon);

    Ok(())
}

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
