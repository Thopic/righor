use anyhow::Result;
use righor::shared::sequence::ExtendedDna;
use righor::shared::{AlignmentParameters, DnaLike, InferenceParameters};
mod common;
use ndarray::Array2;
use righor::shared::errors::ErrorConstantRate;

use righor::shared::DNAMarkovChain;
use righor::shared::ErrorParameters;
use righor::shared::ModelStructure;
use righor::{Dna, AminoAcid};
use ndarray::array;
use righor::EntrySequence;
use righor::Modelable;




#[test]
fn evaluate_cdr3() -> Result<()> {
    let model = common::simple_model_vdj();
    // ATCTACTACTACTGCTCATGCAAATTTTTCGCTTTTGGGGCAGTCTTCGGAGAAACAAAGACTTAT
    // nt:               TGCAAATTTTTCGCTTTTGGGGCAGTCTTC
    //                         TTTTTCGCTTTT
    //                                   GGGGGGCAGTCTTCGGAGAAACAAAGACTTAT
    // ATCTACTACTACTGCTCATGCAAAAAAAAATT
    let dna = Dna::from_string("TGCAAATTTTTCGCTTTTGGGGCAGTCTTC")?;
    let cdr3 = dna.clone().translate()?;
    println!("{}", cdr3.to_string());
    let es = EntrySequence::NucleotideCDR3((cdr3.clone().into(), model.clone().seg_vs, model.clone().seg_js));
    let mut ip = InferenceParameters::default();
    ip.infer_features = false;
    
    let result = model.evaluate(es, &AlignmentParameters::default(), &ip);
    println!("{:?}\n", result);

    
    let es = EntrySequence::NucleotideCDR3((dna.clone().into(), model.clone().seg_vs, model.clone().seg_js));
    let result = model.evaluate(es, &AlignmentParameters::default(), &ip);
    println!("{:?}\n", result);


    let es = EntrySequence::NucleotideSequence(Dna::from_string("ATCTACTACTACTGCTCATGCAAATTTTTCGCTTTTGGGGCAGTCTTCGGAGAAACAAAGACTTAT")?.into());

    let result = model.evaluate(es, &AlignmentParameters::default(), &ip)?;
    println!("{:?}", result);

    
    println!("{}", cdr3.clone().to_dnas().len());
    let mut sum = 0.;
    for nt in &cdr3.to_dnas() {
//	println!("{}", nt.to_string());
	let es = EntrySequence::NucleotideCDR3((nt.clone().into(), model.clone().seg_vs, model.clone().seg_js));
	//println!("SEQ: {}", nt.to_string());
	let result = model.evaluate(es, &AlignmentParameters::default(), &ip)?;
	sum += result.likelihood;
	if result.likelihood > 0. {
	    println!("{} {}", nt.to_string(), result.likelihood);
	}
    }
    println!("{}", sum);
    
    Ok(())
    


}


#[test]
fn evaluate_markov_amino_acid() -> Result<()> {
    let epsilon = 1e-10;
    let mut array =  array![
	    [0.2, 0.5, 0.3, 0.7],
            [0.3, 0.5, 0.3, 0.7],
            [0.4, 0.1, 0.4, 0.01],
            [0.5, 0.8, 0.4, 0.002]];

    let mkc = DNAMarkovChain::new(&array)?;

    let aa = AminoAcid::from_string("WMM")?;
    let seq = ExtendedDna::from_aminoacid(aa.clone());
    let raa = mkc.likelihood_aminoacid(&seq, 0);
    let mut rnt = 0.;
    for seqnt in aa.to_dnas() {
	rnt += mkc.likelihood_dna(&seqnt, 0);
    }
    assert!((raa - rnt).abs() < epsilon);


    let aa = AminoAcid::from_string("CFW")?;
    let seq = ExtendedDna::from_aminoacid(aa.clone());
    let raa = mkc.likelihood_aminoacid(&seq, 0);
    let mut rnt = 0.;
    for seqnt in aa.to_dnas() {
	rnt += mkc.likelihood_dna(&seqnt, 0);
    }
    assert!((raa - rnt).abs() < epsilon);

    let aa = AminoAcid::from_string("WML")?;
    let seq = ExtendedDna::from_aminoacid(aa.clone());
    let raa = mkc.likelihood_aminoacid(&seq, 0);
    let mut rnt = 0.;
    for seqnt in aa.to_dnas() {
	rnt += mkc.likelihood_dna(&seqnt, 0);
    }
    assert!((raa - rnt).abs() < epsilon);

    let aa = AminoAcid::from_string("LML")?;
    let mut seq = ExtendedDna::from_aminoacid(aa.clone());
    seq = seq.extract_subsequence(1, 9);
    let raa = mkc.likelihood_aminoacid(&seq, 1);
    let mut rnt = 0.;
    for seqnt in aa.to_dnas() {
	if seqnt.seq[0] == b'C' {
	    rnt += mkc.likelihood_dna(&seqnt.extract_subsequence(1, 9), 1);
	}
    }

    assert!((raa - rnt).abs() < epsilon);

    let aa = AminoAcid::from_string("L")?;
    let seq = ExtendedDna::from_aminoacid(aa.clone());
    let raa = mkc.likelihood_aminoacid(&seq, 0);
    let mut rnt = 0.;
    for seqnt in aa.to_dnas() {
	rnt += mkc.likelihood_dna(&seqnt, 0);
    }
    assert!((raa - rnt).abs() < epsilon);


    
//    assert!((mkc.likelihood_aminoacid(&seq, 0) - 1.52587890625e-05).abs() < epsilon); // 1/4^8

    // let seq = ExtendedDna::from_aminoacid(AminoAcid::from_string("CFW")?); // two Y
    // assert!((mkc.likelihood_aminoacid(&seq) - 6.103515625e-05).abs() < epsilon); // 1/4^7

    // let seq = ExtendedDna::from_aminoacid(AminoAcid::from_string("WMP")?); // end with N

    // assert!((mkc.likelihood_aminoacid(&seq) - 6.103515625e-05).abs() < epsilon); // 1/4^7

    // let seq = ExtendedDna::from_aminoacid(AminoAcid::from_string("WML")?); // 6 for the last one
    // assert!((mkc.likelihood_aminoacid(&seq) - 9.1552734375e-05).abs() < epsilon); // 6/4^8

    // let mut seq = ExtendedDna::from_aminoacid(AminoAcid::from_string("PMW")?);
    // seq = seq.extract_subsequence(2, 9); // Start with N
    // println!("{} {}", seq.to_dna().to_string(), mkc.likelihood_aminoacid(&seq));
    // assert!((mkc.likelihood_aminoacid(&seq) - 0.0009765625).abs() < epsilon); // 1/4^6


    // let mut seq = ExtendedDna::from_aminoacid(AminoAcid::from_string("SM")?);
    // seq = seq.extract_subsequence(1, 6); 
    // println!("{} {}", seq.to_dna().to_string(), mkc.likelihood_aminoacid(&seq));
    
    //assert!((mkc.likelihood_aminoacid(&seq) - 0.000244140625).abs() < epsilon); // 1/4^6


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
