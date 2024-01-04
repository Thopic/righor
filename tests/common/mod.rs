use ihor::vdj;
use ndarray::array;

/// Create a simple model with only 3 genes (one V, one D, one J)
pub fn simple_model_vdj() -> vdj::Model {
    let gv1 = ihor::Gene {
        name: "V1".to_string(),
        seq: ihor::Dna::from_string("TGCTCAGTAAAAAAAAAA").unwrap(), // need to start with a cystein
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(0),
    };
    let gv2 = ihor::Gene {
        name: "V2".to_string(),
        seq: ihor::Dna::from_string("TGCTCCAAAAGTGGGGGG").unwrap(), // need to start with a cystein
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(0),
    };
    let gv3 = ihor::Gene {
        name: "V2".to_string(),
        seq: ihor::Dna::from_string("TGGGTCAAAAGTCCCCCC").unwrap(), // need to start with a cystein
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(0),
    };
    let gj1 = ihor::Gene {
        name: "J1".to_string(),
        seq: ihor::Dna::from_string("GGGGGGCAGTCAGT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(11), // the cdr3 pos is defined before the last amino-acid, so 13-3=10
    };
    let gj2 = ihor::Gene {
        name: "J2".to_string(),
        seq: ihor::Dna::from_string("CCGCCCCACACAGT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(11), // the cdr3 pos is defined before the last amino-acid, so 13-3=10
    };
    let gj3 = ihor::Gene {
        name: "J2".to_string(),
        seq: ihor::Dna::from_string("AATACCCACACAGT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(11), // the cdr3 pos is defined before the last amino-acid, so 13-3=10
    };

    let gd1 = ihor::Gene {
        name: "D1".to_string(),
        seq: ihor::Dna::from_string("TTTTTCGCTTTT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: None,
    };
    let gd2 = ihor::Gene {
        name: "D2".to_string(),
        seq: ihor::Dna::from_string("AAAAACGCAAAA").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: None,
    };

    let mut model = ihor::vdj::Model {
        seg_vs: vec![gv1, gv2, gv3],
        seg_js: vec![gj1, gj2, gj3],
        seg_ds: vec![gd1, gd2],
        p_v: array![0.5, 0.25, 0.25],
        p_dj: array![[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]],
        p_ins_vd: array![0.4, 0.2, 0.1, 0.01, 0.001], // up to 4 insertions
        p_ins_dj: array![0.4, 0.2, 0.1, 0.01, 0.001, 0.0001], // up to 5 insertions
        p_del_v_given_v: array![
            [0.04, 0.04, 0.04],
            [0.2, 0.2, 0.2],
            [1., 1., 1.],
            [0.1, 0.1, 0.1],
            [0.01, 0.01, 0.01],
            [0.001, 0.001, 0.001],
            [0.0001, 0.0001, 0.0001]
        ],
        p_del_j_given_j: array![
            [0.04, 0.04, 0.04],
            [0.2, 0.2, 0.2],
            [2., 2., 2.],
            [0.1, 0.1, 0.1],
            [0.01, 0.01, 0.01],
            [0.001, 0.001, 0.001],
            [0.0001, 0.0001, 0.0001]
        ],
        p_del_d3_del_d5: array![
            [
                [1.6e-03, 1.6e-03],
                [8.0e-03, 8.0e-03],
                [4.0e-02, 4.0e-02],
                [4.0e-03, 4.0e-03],
                [4.0e-04, 4.0e-04],
                [4.0e-05, 4.0e-05],
                [4.0e-06, 4.0e-06]
            ],
            [
                [8.0e-03, 8.0e-03],
                [4.0e-02, 4.0e-02],
                [2.0e-01, 2.0e-01],
                [2.0e-02, 2.0e-02],
                [2.0e-03, 2.0e-03],
                [2.0e-04, 2.0e-04],
                [2.0e-05, 2.0e-05]
            ],
            [
                [4.0e-02, 4.0e-02],
                [2.0e-01, 2.0e-01],
                [1.0e+00, 1.0e+00],
                [1.0e-01, 1.0e-01],
                [1.0e-02, 1.0e-02],
                [1.0e-03, 1.0e-03],
                [1.0e-04, 1.0e-04]
            ],
            [
                [4.0e-03, 4.0e-03],
                [2.0e-02, 2.0e-02],
                [1.0e-01, 1.0e-01],
                [1.0e-02, 1.0e-02],
                [1.0e-03, 1.0e-03],
                [1.0e-04, 1.0e-04],
                [1.0e-05, 1.0e-05]
            ],
            [
                [4.0e-04, 4.0e-04],
                [2.0e-03, 2.0e-03],
                [1.0e-02, 1.0e-02],
                [1.0e-03, 1.0e-03],
                [1.0e-04, 1.0e-04],
                [1.0e-05, 1.0e-05],
                [1.0e-06, 1.0e-06]
            ],
            [
                [4.0e-05, 4.0e-05],
                [2.0e-04, 2.0e-04],
                [1.0e-03, 1.0e-03],
                [1.0e-04, 1.0e-04],
                [1.0e-05, 1.0e-05],
                [1.0e-06, 1.0e-06],
                [1.0e-07, 1.0e-07]
            ],
            [
                [4.0e-06, 4.0e-06],
                [2.0e-05, 2.0e-05],
                [1.0e-04, 1.0e-04],
                [1.0e-05, 1.0e-05],
                [1.0e-06, 1.0e-06],
                [1.0e-07, 1.0e-07],
                [1.0e-08, 1.0e-08]
            ]
        ],
        markov_coefficients_vd: array![
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ],
        markov_coefficients_dj: array![
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.2, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ],
        first_nt_bias_ins_vd: array![0.25, 0.25, 0.25, 0.25],
        first_nt_bias_ins_dj: array![0.25, 0.25, 0.25, 0.25],
        range_del_v: (-2, 4),
        range_del_j: (-2, 4),
        range_del_d3: (-2, 4),
        range_del_d5: (-2, 4),
        error_rate: 0.,
        ..Default::default()
    };
    model.initialize().unwrap();
    model
}

/// Create a simple model with only 6 genes (two V, two D, two J), no insertions
pub fn simple_model_vdj_no_ins() -> vdj::Model {
    let gv1 = ihor::Gene {
        name: "V1".to_string(),
        seq: ihor::Dna::from_string("TGCTCAGTAAAAAAAAAA").unwrap(), // need to start with a cystein
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(0),
    };
    let gv2 = ihor::Gene {
        name: "V2".to_string(),
        seq: ihor::Dna::from_string("TGCTCCAAAAGTGGGGGG").unwrap(), // need to start with a cystein
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(0),
    };
    let gv3 = ihor::Gene {
        name: "V2".to_string(),
        seq: ihor::Dna::from_string("TGGGTCAAAAGTCCCCCC").unwrap(), // need to start with a cystein
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(0),
    };
    let gj1 = ihor::Gene {
        name: "J1".to_string(),
        seq: ihor::Dna::from_string("GGGGGGCAGTCAGT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(11), // the cdr3 pos is defined before the last amino-acid, so 13-3=10
    };
    let gj2 = ihor::Gene {
        name: "J2".to_string(),
        seq: ihor::Dna::from_string("CCGCCCCACACAGT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(11), // the cdr3 pos is defined before the last amino-acid, so 13-3=10
    };
    let gj3 = ihor::Gene {
        name: "J2".to_string(),
        seq: ihor::Dna::from_string("AATACCCACACAGT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(11), // the cdr3 pos is defined before the last amino-acid, so 13-3=10
    };

    let gd1 = ihor::Gene {
        name: "D1".to_string(),
        seq: ihor::Dna::from_string("TTTTTCGCTTTT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: None,
    };
    let gd2 = ihor::Gene {
        name: "D2".to_string(),
        seq: ihor::Dna::from_string("AAAAACGCAAAA").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: None,
    };

    let mut model = ihor::vdj::Model {
        seg_vs: vec![gv1, gv2, gv3],
        seg_js: vec![gj1, gj2, gj3],
        seg_ds: vec![gd1, gd2],
        p_v: array![0.5, 0.25, 0.25],
        p_dj: array![[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]],
        p_ins_vd: array![1.], // 0 insertions
        p_ins_dj: array![1.],
        p_del_v_given_v: array![
            [0.04, 0.04, 0.04],
            [0.2, 0.2, 0.2],
            [1., 1., 1.],
            [0.1, 0.1, 0.1],
            [0.01, 0.01, 0.01],
            [0.001, 0.001, 0.001],
            [0.0001, 0.0001, 0.0001]
        ],
        p_del_j_given_j: array![
            [0.04, 0.04, 0.04],
            [0.2, 0.2, 0.2],
            [2., 2., 2.],
            [0.1, 0.1, 0.1],
            [0.01, 0.01, 0.01],
            [0.001, 0.001, 0.001],
            [0.0001, 0.0001, 0.0001]
        ],
        p_del_d3_del_d5: array![
            [
                [1.6e-03, 1.6e-03],
                [8.0e-03, 8.0e-03],
                [4.0e-02, 4.0e-02],
                [4.0e-03, 4.0e-03],
                [4.0e-04, 4.0e-04],
                [4.0e-05, 4.0e-05],
                [4.0e-06, 4.0e-06]
            ],
            [
                [8.0e-03, 8.0e-03],
                [4.0e-02, 4.0e-02],
                [2.0e-01, 2.0e-01],
                [2.0e-02, 2.0e-02],
                [2.0e-03, 2.0e-03],
                [2.0e-04, 2.0e-04],
                [2.0e-05, 2.0e-05]
            ],
            [
                [4.0e-02, 4.0e-02],
                [2.0e-01, 2.0e-01],
                [1.0e+00, 1.0e+00],
                [1.0e-01, 1.0e-01],
                [1.0e-02, 1.0e-02],
                [1.0e-03, 1.0e-03],
                [1.0e-04, 1.0e-04]
            ],
            [
                [4.0e-03, 4.0e-03],
                [2.0e-02, 2.0e-02],
                [1.0e-01, 1.0e-01],
                [1.0e-02, 1.0e-02],
                [1.0e-03, 1.0e-03],
                [1.0e-04, 1.0e-04],
                [1.0e-05, 1.0e-05]
            ],
            [
                [4.0e-04, 4.0e-04],
                [2.0e-03, 2.0e-03],
                [1.0e-02, 1.0e-02],
                [1.0e-03, 1.0e-03],
                [1.0e-04, 1.0e-04],
                [1.0e-05, 1.0e-05],
                [1.0e-06, 1.0e-06]
            ],
            [
                [4.0e-05, 4.0e-05],
                [2.0e-04, 2.0e-04],
                [1.0e-03, 1.0e-03],
                [1.0e-04, 1.0e-04],
                [1.0e-05, 1.0e-05],
                [1.0e-06, 1.0e-06],
                [1.0e-07, 1.0e-07]
            ],
            [
                [4.0e-06, 4.0e-06],
                [2.0e-05, 2.0e-05],
                [1.0e-04, 1.0e-04],
                [1.0e-05, 1.0e-05],
                [1.0e-06, 1.0e-06],
                [1.0e-07, 1.0e-07],
                [1.0e-08, 1.0e-08]
            ]
        ],
        markov_coefficients_vd: array![
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ],
        markov_coefficients_dj: array![
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.2, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ],
        first_nt_bias_ins_vd: array![0.25, 0.25, 0.25, 0.25],
        first_nt_bias_ins_dj: array![0.25, 0.25, 0.25, 0.25],
        range_del_v: (-2, 4),
        range_del_j: (-2, 4),
        range_del_d3: (-2, 4),
        range_del_d5: (-2, 4),
        error_rate: 0.1,
        ..Default::default()
    };
    model.initialize().unwrap();
    model
}

/// Create a simple model with only 3 genes (one V, one D, one J) and no deletions
pub fn simple_model_vdj_no_deletions() -> vdj::Model {
    let gv = ihor::Gene {
        name: "V1".to_string(),
        seq: ihor::Dna::from_string("TGCTCAGTAAAAAAAAAA").unwrap(), // need to start with a cystein
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(0),
    };
    let gj = ihor::Gene {
        name: "J1".to_string(),
        seq: ihor::Dna::from_string("GGGGGGCAGTCAGT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: Some(11), // the cdr3 pos is defined before the last amino-acid, so 13-3=10
    };
    let gd = ihor::Gene {
        name: "D1".to_string(),
        seq: ihor::Dna::from_string("TTTTTTTTTTTT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        cdr3_pos: None,
    };

    let mut model = ihor::vdj::Model {
        seg_vs: vec![gv],
        seg_js: vec![gj],
        seg_ds: vec![gd],
        p_v: array![1.],
        p_dj: array![[1.]],
        p_ins_vd: array![0.1, 0.01, 0.002, 0.0001, 0.00005], // 0, 1, 2, 3, 4 insertions
        p_ins_dj: array![0.1, 0.01, 0.002, 0.0001, 0.00005],
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
            [0.25, 0.25, 0.2, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ],
        first_nt_bias_ins_vd: array![0.25, 0.25, 0.25, 0.25],
        first_nt_bias_ins_dj: array![0.25, 0.25, 0.25, 0.25],
        range_del_v: (0, 1),
        range_del_j: (0, 1),
        range_del_d3: (0, 1),
        range_del_d5: (0, 1),
        ..Default::default()
    };
    model.initialize().unwrap();
    model
}

pub fn inference_parameters_default() -> ihor::InferenceParameters {
    ihor::InferenceParameters {
        min_log_likelihood: -600.0,
        nb_best_events: 10,
        evaluate: true,
    }
}

pub fn alignment_parameters_default() -> ihor::AlignmentParameters {
    ihor::AlignmentParameters {
        min_score_v: 10,
        min_score_j: 10,
        max_error_d: 100,
    }
}
