use ndarray::array;
use righor::shared::errors::ErrorConstantRate;
use righor::shared::ErrorParameters;
use righor::vdj;
use righor::Modelable;

#[cfg(test)]
#[allow(dead_code)]
pub fn simple_model_vdj() -> vdj::Model {
    let gv1 = righor::Gene {
        name: "V1".to_string(),
        seq: righor::Dna::from_string("TGCTCATGCAAAAAAAAA").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        is_functional: true,
        cdr3_pos: Some(6),
    };
    // TGCTCATGCAAAAAAGGAGGCTTTTCTCCCTGTAGTGGGAGGGAGTTAGGTGAGACACAAGGACCTCT
    // TGCTCATGCAAAAAAAAA   TTTTTCGCTTTT   GGGGGGCAGTCAGAGGAGAAACAAAGACTTAT
    let gj1 = righor::Gene {
        name: "J1".to_string(),
        seq: righor::Dna::from_string("GGGGGGCAGTCAGAGGAGAAACAAAGACTTAT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        is_functional: true,
        cdr3_pos: Some(11),
    };
    let gd1 = righor::Gene {
        name: "D1".to_string(),
        seq: righor::Dna::from_string("TTTTTCGCTTTT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        is_functional: true,
        cdr3_pos: None,
    };

    let gd2 = righor::Gene {
        name: "D2".to_string(),
        seq: righor::Dna::from_string("TTAAAACGCAATT").unwrap(),
        seq_with_pal: None,
        functional: "(F)".to_string(),
        is_functional: true,
        cdr3_pos: None,
    };

    let mut model = righor::vdj::Model {
        seg_vs: vec![gv1],
        seg_js: vec![gj1],
        seg_ds: vec![gd1, gd2],
        p_vdj: array![[[0.6], [0.4]]],
        p_ins_vd: array![0.401, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05],
        p_ins_dj: array![0.402, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05],
        p_del_v_given_v: array![[0.1005], [0.2], [0.4], [0.1], [0.05], [0.1], [0.05],],
        p_del_j_given_j: array![[0.1], [0.2], [0.4], [0.1], [0.05], [0.1], [0.05],],
        p_del_d5_del_d3: array![
            [
                [0.05, 0.05],
                [0.1, 0.1],
                [0.05, 0.05],
                [0.324, 0.324],
                [0.02, 0.02]
            ],
            [
                [0.0508, 0.0508],
                [0.1, 0.1],
                [0.0512, 0.0512],
                [0.3, 0.3],
                [0.02, 0.02]
            ],
            [
                [0.05, 0.05],
                [0.35, 0.35],
                [0.05, 0.05],
                [0.3, 0.3],
                [0.02, 0.02]
            ],
        ],
        markov_coefficients_vd: array![
            [0.259, 0.25, 0.25, 0.254],
            [0.25, 0.25, 0.251, 0.25],
            [0.25, 0.252, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ],
        markov_coefficients_dj: array![
            [0.2501, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.2592, 0.25, 0.25],
            [0.25, 0.25, 0.2533, 0.2512]
        ],
        range_del_v: (-2, 4),
        range_del_j: (-2, 4),
        range_del_d3: (-1, 1),
        range_del_d5: (-1, 3),
        error: ErrorParameters::ConstantRate(ErrorConstantRate::new(0.)),
        ..Default::default()
    };

    model.initialize().unwrap();
    model
}
