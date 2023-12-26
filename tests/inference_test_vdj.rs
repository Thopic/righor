use ihor;
use ndarray::array;

#[test]
fn test_most_likely_no_del() {
    // all the tests that relate to the inference of the most likely recombination scenario.
    let gv = ihor::Gene {
        name: "V1".to_string(),
        seq: ihor::Dna::from_string("CAGTCAGTAAAAAAAAAA").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(0),
    };
    let gj = ihor::Gene {
        name: "J1".to_string(),
        seq: ihor::Dna::from_string("GGGGGGCAGTCAGT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(0),
    };
    let gd = ihor::Gene {
        name: "D1".to_string(),
        seq: ihor::Dna::from_string("TTTTTTTTTTTT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(0),
    };

    let mut model = ihor::vdj::Model {
        seg_vs: vec![gv],
        seg_js: vec![gj],
        seg_ds: vec![gd],
        p_v: array![1.],
        p_dj: array![[1.]],
        p_ins_vd: array![0.5, 0.25, 0.125, 0.125], // 0, 1, 2, 3 insertions
        p_ins_dj: array![0.5, 0.25, 0.125, 0.125],
        p_del_v_given_v: array![[1.]],
        p_del_j_given_j: array![[1.]],
        p_del_d3_del_d5: array![[[1.]]],
        markov_coefficients_vd: array![
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ],
        markov_coefficients_dj: array![
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.2, 0.3], // G->T more likely
            [0.25, 0.25, 0.25, 0.25]
        ],
        first_nt_bias_ins_vd: array![0.25, 0.25, 0.25, 0.25],
        first_nt_bias_ins_dj: array![0.25, 0.25, 0.25, 0.25],
        range_del_v: (0, 0),
        range_del_j: (0, 0),
        range_del_d3: (0, 0),
        range_del_d5: (0, 0),
        ..Default::default()
    };
    model.sanitize_genes().unwrap();

    let if_params = ihor::InferenceParameters {
        min_likelihood_error: 1e-40,
        min_likelihood: 1e-60,
    };

    let al_params = ihor::AlignmentParameters {
        min_score_v: 10,
        min_score_j: 10,
        max_error_d: 8,
    };

    // No insertions or deletions
    let myseq = ihor::Dna::from_string("CAGTCAGTAAAAAAAAAATTTTTTTTTTTTGGGGGGCAGTCAGT");
    let seq_aligned = model.align_sequence(myseq.unwrap(), &al_params);
    let result = model
        .most_likely_recombinations(&seq_aligned.unwrap(), 5, &if_params)
        .unwrap();
    assert!(result.clone()[0].1.deld3 == 0);
    assert!(result.clone()[0].1.insvd == ihor::Dna::from_string("").unwrap());

    // Add an insertion (VD)
    let myseq = ihor::Dna::from_string("CAGTCAGTAAAAAAAAAACTTTTTTTTTTTTGGGGGGCAGTCAGT");
    let seq_aligned = model.align_sequence(myseq.unwrap(), &al_params);
    let result = model
        .most_likely_recombinations(&seq_aligned.unwrap(), 5, &if_params)
        .unwrap();
    assert!(result.clone()[0].1.deld3 == 0);
    assert!(result.clone()[0].1.insvd == ihor::Dna::from_string("C").unwrap());

    // Add an insertion (DJ)
    let myseq = ihor::Dna::from_string("CAGTCAGTAAAAAAAAAATTTTTTTTTTTTAGGGGGGCAGTCAGT");
    let seq_aligned = model.align_sequence(myseq.unwrap(), &al_params);
    let result = model
        .most_likely_recombinations(&seq_aligned.unwrap(), 5, &if_params)
        .unwrap();
    assert!(result.clone()[0].1.deld3 == 0);
    assert!(result.clone()[0].1.insvd == ihor::Dna::from_string("").unwrap());
    assert!(result.clone()[0].1.insdj == ihor::Dna::from_string("A").unwrap());

    // Add two insertions (VD/DJ)
    let myseq = ihor::Dna::from_string("CAGTCAGTAAAAAAAAAAGCTTTTTTTTTTTTCGGGGGGGCAGTCAGT");
    let seq_aligned = model.align_sequence(myseq.unwrap(), &al_params);
    let result = model
        .most_likely_recombinations(&seq_aligned.unwrap(), 5, &if_params)
        .unwrap();
    assert!(result.clone()[0].1.deld3 == 0);
    assert!(result.clone()[0].1.insvd == ihor::Dna::from_string("GC").unwrap());
    assert!(result.clone()[0].1.insdj == ihor::Dna::from_string("CG").unwrap());

    // Add two insertions. Could be GT-TT or GTT-T or G-TTT. 2 and 3 are more likely (with
    // the first one having an edge because G->T is more likely than T->T)
    let myseq = ihor::Dna::from_string("CAGTCAGTAAAAAAAAAAGTTTTTTTTTTTTTTTGGGGGGCAGTCAGT");
    let seq_aligned = model.align_sequence(myseq.unwrap(), &al_params);
    let result = model
        .most_likely_recombinations(&seq_aligned.unwrap(), 5, &if_params)
        .unwrap();
    println!("{:?}", result);

    assert!(result.clone()[2].1.insvd == ihor::Dna::from_string("GT").unwrap());
    assert!(result.clone()[2].1.insdj == ihor::Dna::from_string("TT").unwrap());
    assert!(result.clone()[0].1.insvd == ihor::Dna::from_string("GTT").unwrap());
    assert!(result.clone()[0].1.insdj == ihor::Dna::from_string("T").unwrap());
    assert!(result.clone()[1].1.insvd == ihor::Dna::from_string("G").unwrap());
    assert!(result.clone()[1].1.insdj == ihor::Dna::from_string("TTT").unwrap());
}

#[test]
fn test_most_likely_no_ins() {
    // all the tests that relate to the inference of the most likely recombination scenario.
    let gv = ihor::Gene {
        name: "V1".to_string(),
        seq: ihor::Dna::from_string("CAGTCAGTAAAAAAAAAA").unwrap(),
        seq_with_pal: None,
        functional: "".to_string(),
        cdr3_pos: Some(0),
    };
    let gj = ihor::Gene {
        name: "J1".to_string(),
        seq: ihor::Dna::from_string("GGGGGGCAGTCAGT").unwrap(),
        seq_with_pal: None,
        functional: "".to_string(),
        cdr3_pos: Some(0),
    };
    let gd = ihor::Gene {
        name: "D1".to_string(),
        seq: ihor::Dna::from_string("TTTTTTTTTTTT").unwrap(),
        seq_with_pal: None,
        functional: "".to_string(),
        cdr3_pos: Some(0),
    };

    let mut model = ihor::vdj::Model {
        seg_vs: vec![gv],
        seg_js: vec![gj],
        seg_ds: vec![gd],
        p_v: array![1.],
        p_dj: array![[0.5, 0.5,]],
        p_ins_vd: array![1.],
        p_ins_dj: array![1.],
        p_del_v_given_v: array![[0.6], [0.2], [0.1], [0.05], [0.05]], // -1, 0, 1, 2, 3
        p_del_j_given_j: array![[0.6], [0.2], [0.1], [0.05], [0.05]],
        p_del_d3_del_d5: array![[[1.]]],
        markov_coefficients_vd: array![
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ],
        markov_coefficients_dj: array![
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ],
        first_nt_bias_ins_vd: array![0.25, 0.25, 0.25, 0.25],
        first_nt_bias_ins_dj: array![0.25, 0.25, 0.25, 0.25],
        range_del_v: (-1, 3),
        range_del_j: (-1, 3),
        range_del_d3: (0, 0),
        range_del_d5: (0, 0),
        ..Default::default()
    };
    model.sanitize_genes().unwrap();

    let if_params = ihor::InferenceParameters {
        min_likelihood_error: 1e-40,
        min_likelihood: 1e-60,
    };

    let al_params = ihor::AlignmentParameters {
        min_score_v: 10,
        min_score_j: 10,
        max_error_d: 8,
    };

    // No insertions or deletions
    let myseq = ihor::Dna::from_string("CAGTCAGTAAAAAAAAAATTTTTTTTTTTTGGGGGGCAGTCAGT");
    let seq_aligned = model.align_sequence(myseq.unwrap(), &al_params).unwrap();
    let result = model
        .most_likely_recombinations(&seq_aligned, 5, &if_params)
        .unwrap();
    println!("{:?}", result);
    assert!(result.clone()[0].1.delv == 1);
    assert!(result.clone()[0].1.delj == 1);
    assert!(result.clone()[0].1.insvd == ihor::Dna::from_string("").unwrap());

    // 1 palindromic insertion on V
    let myseq = ihor::Dna::from_string("CAGTCAGTAAAAAAAAAATTTTTTTTTTTTTGGGGGGCAGTCAGT");
    let seq_aligned = model.align_sequence(myseq.unwrap(), &al_params).unwrap();
    let result = model
        .most_likely_recombinations(&seq_aligned, 5, &if_params)
        .unwrap();
    println!("{:?}", result);
    assert!(result.clone()[0].1.delv == 0);
    assert!(result.clone()[0].1.delj == 1);
    assert!(result.clone()[0].1.insvd == ihor::Dna::from_string("").unwrap());

    // 2 deletion on V, one palindromic insert on J
    let myseq = ihor::Dna::from_string("CAGTCAGTAAAAAAAATTTTTTTTTTTTCGGGGGGCAGTCAGT");
    let seq_aligned = model.align_sequence(myseq.unwrap(), &al_params).unwrap();
    let result = model
        .most_likely_recombinations(&seq_aligned, 5, &if_params)
        .unwrap();
    println!("{:?}", result);
    assert!(result.clone()[0].1.delv == 3);
    assert!(result.clone()[0].1.delj == 0);
    assert!(result.clone()[0].1.insvd == ihor::Dna::from_string("").unwrap());
}
