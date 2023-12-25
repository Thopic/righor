extern crate ihor;
use ihor::*;

#[test]
fn test_most_likely() {
    // all the tests that relate to the inference of the most likely recombination scenario.
    let gV = ihor::GeneV {
        name: "V1",
        seq: ihor::Dna::from_string("CAGTCAGTAAAAAAAAAA"),
        seq_with_pal: None,
        functional: "(F)",
        cdr3_pos: 0,
    };
    let gJ = ihor::GeneJ {
        name: "J1",
        seq: ihor::Dna::from_string("GGGGGGCAGTCAGT"),
        seq_with_pal: None,
        functional: "(F)",
        cdr3_pos: 0,
    };
    let gD = ihor::GeneD {
        name: "D1",
        seq: ihor::Dna::from_string("TTTTTTTTTTTT"),
        seq_with_pal: None,
        functional: "(F)",
        cdr3_pos: 0,
    };

    let model = ModelVDJ {
        seq_vs: vec![gV],
        seq_js: vec![gJ],
        seq_ds: vec![gJ],
        p_v: array![1.],
        p_dj: array![[1.]],
        p_ins_vd: array![0.5, 0.25, 0.25], // 0 ins, 1 ins, 2 ins
        p_ins_dj: array![0.5, 0.25, 0.25],
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
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ],
        first_nt_bias_insvd: array![0.25, 0.25, 0.25, 0.25],
        first_nt_bias_insdj: array![0.25, 0.25, 0.25, 0.25],
        range_del_v: (0, 0),
        range_del_j: (0, 0),
        range_del_d3: (0, 0),
        range_del_d5: (0, 0),
        ..Default::default()
    };
    model.sanitize_genes();

    let ifp = InferenceParameters {
        min_likelihood_error: 1e-40,
        min_likelihood: 1e-60,
    };

    let myseq = ihor::Dna::from_string("CAGTCAGTAAAAAAAAAATTTTTTTTTTTTGGGGGGCAGTCAGT");
    let result = most_likely_recombinations(5, myseq, model, ifp);
    assert!(result[0].1.deld3 == 0);
}
