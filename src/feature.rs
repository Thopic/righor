// A sequence is a collection of feature, each feature has some probability
// When we iterate on the model we update the probability of each feature

use crate::sequence::SequenceVDJ;

pub struct InferenceParams {
    min_likelihood: f64,
}

#[derive(Default, Clone, Debug)]
pub struct FeaturesVDJ {
    v: Array1<f64>,
    delv: Array2<f64>,
    dj: Array2<f64>,
    delj: Array2<f64>,
    deld: Array3<f64>,
    insvd: Array1<f64>,
    insdj: Array1<f64>,
    first_nt_bias_vd: Array1<f64>,
    markov_coefficients_vd: Array2<f64>,
    first_nt_bias_dj: Array1<f64>,
    markov_coefficients_dj: Array2<f64>,
}

pub struct MarginalsVDJ {
    // the marginals used during the inference
    marginals: CategoricalFeaturesVDJ,

    // this variable store the evolving marginals as they
    // are iterated upon (in particular it's not expected to
    // be normalized)
    dirty_marginals: CategoricalFeaturesVDJ,
}

impl FeaturesVDJ {
    fn normalize(&mut self) -> Result<(), Box<dyn Error>> {
        // Normalize the arrays
        // Because of the way the original code is set up
        // some distributions are identically 0 (p(delv|v=v1) if
        // v1 has probability 0 for example)
        // We modify these probability to make them uniform
        // (and avoid issues like log(0))
        FeaturesVDJ {
            v: v.normalize_distribution()?,
            dj: dj.normalize_distribution(Axis(0))?,
            dj: dj.normalize_distribution(Axis(1))?,
            delv: delv.normalize_distribution(Axis(0))?,
            delj: delj.normalize_distribution(Axis(0))?,
            deld: deld.normalize_distribution(Axis(0))?,
            deld: deld.normalize_distribution(Axis(1))?,
            first_nt_bias_vd: first_nt_bias_vd.normalize_distribution()?,
            first_nt_bias_dj: first_nt_bias_dj.normalize_distribution()?,
            markov_coefficient_vd: markov_coefficients_vd.normalize_distribution(Axis(1))?,
            markov_coefficient_dj: markov_coefficients_dj.normalize_distribution(Axis(1))?,
        }
    }
}

#[derive(Default, Clone, Debug)]
impl MarginalsVDJ {
    fn new(model: Option<&ModelVDJ>) -> MarginalsVDJ {
        let m: MarginalsVDJ = Default::default();

        m.marginals = Default::default;
        m.marginals.v = model.p_v.clone();
        m.marginals.dj = model.p_dj.clone();
        m.marginals.delj = model.p_del_j_given_j.clone();
        m.marginals.deld = model.p_del_v_given_v.clone();
        m.marginals.insvd = model.p_del_d3_del_d5.clone();
        m.marginals.insdj = model.p_ins_vd.clone();
        // in case, normalize the original marginals
        m.marginals.normalize();

        // Dirty marginals starts empty by default
        m.dirty_marginals = Default::default;
        m.dirty_marginals.v = Array1::zeros(m.marginals.v.dim());
        m.dirty_marginals.dj = Array2::zeros(m.marginals.dj.dim());
        m.dirty_marginals.delj = Array2::zeros(m.marginals.delj.dim());
        m.dirty_marginals.deld = Array3::zeros(m.marginals.deld.dim());
        m.dirty_marginals.insvd = Array1::zeros(m.marginals.insvd.dim());
        m.dirty_marginals.insdj = Array1::zeros(m.marginals.insdj.dim());
    }

    fn likelihood_v(&self, vi: usize) -> f64 {
        self.marginals.v[vi]
    }
    fn likelihood_delv(&self, dv: usize, vi: usize, seq: &SequenceVDJ) -> f64 {
        // needs to add the effect of the error
        self.marginals.delv[[dv, v]]
    }
    fn likelihood_dj(&self, di: usize, ji: usize) -> f64 {
        // needs to add the effect of the error
        self.marginals.dj[[di, ji]]
    }
    fn likelihood_delj(&self, dj: usize, ji: usize, seq: &SequenceVDJ) -> f64 {
        self.marginals.delj[[dj, ji]]
    }
    fn likelihood_deld(&self, dd3: usize, dd5: usize, di: usize, seq: &SequenceVDJ) -> (f64, f64) {
        // needs to add the effect of the error
        self.marginals.deld[[dd3, dd5, di]]
    }

    fn likelihood_nb_ins_vd(&self, seq: &Dna) -> f64 {
        self.marginals_insvd[[seq.len()]]
    }
    fn likelihood_ins_vd(&self, seq: &Dna) -> f64 {
        likelihood_markov(&self.first_nt_bias_vd, &self.markov_coefficients_vd, seq)
    }
    fn likelihood_nb_ins_dj(&self, seq: &Dna) -> f64 {
        self.marginals_insdj[seq.len()]
    }
    fn likelihood_ins_dj(&self, seq: &Dna) -> f64 {
        likelihood_markov(&self.first_nt_bias_dj, &self.markov_coefficients_dj, seq)
    }

    fn dirty_update(
        &mut self,
        v: usize,
        d: usize,
        j: usize,
        delv: usize,
        delj: usize,
        deld3: usize,
        deld5: usize,
        nb_insvj: usize,
        nb_insdj: usize,
        insvd: &Dna,
        insdj: &Dna,
        likelihood: f64,
    ) {
        self.dirty_marginals.v[[v]] += likelihood;
        self.dirty_marginals.dj[[d, j]] += likelihood;
        self.dirty_marginals.delv[[delv, v]] += likelihood;
        self.dirty_marginals.delj[[delj, j]] += likelihood;
        self.dirty_marginals.deld[[deld3, deld5, d]] += likelihood;
        self.dirty_marginals.insvd[[nb_insvd]] += likelihood;
        self.dirty_marginals.insdj[[nb_insdj]] += likelihood;
        update_markov_probas(
            self.dirty_marginals.first_nt_bias_vd,
            self.dirty_marginals.markov_coefficients_vd,
            insvd,
            likelihood,
        );
        update_markov_probas(
            self.dirty_marginals.first_nt_bias_dj,
            self.dirty_marginals.markov_coefficients_dj,
            insdj,
            likelihood,
        );
    }

    fn update(&mut self) {
        // normalize the dirty marginals and move them to the
        // current marginals
        m.marginals = dirty_marginals.normalize();
    }

    fn infer_features(
        sequence: SequenceVDJ,
        &mut marginals: MarginalsVDJ,
        inference_params: InferenceParams,
    ) {
        for v in sequence.v_genes {
            let l_v = marginals.likelihood_v(v.index);
            for delv in 0..model.max_del_v {
                let l_delv = marginals.likelihood_delv(delv, v.index, &sequence);
                for j in sequences.j_genes {
                    for d in sequences.d_genes {
                        l_dj = marginals.likelihood_dj(d.index, j.index);
                        for delj in 0..model.max_del_j {
                            l_delj = marginals.likelihood_delj(delj, j.index, &sequence);

                            if l_delj * l_dj * l_delv * l_v < inference_params.min_likelihood {
                                continue;
                            }

                            for deld3 in 0..model.max_deld3 {
                                for deld5 in 0..model.max_deld5 {
                                    let (l_deld3, l_deld5) =
                                        self.likelihood_deld(deld3, deld5, d, &sequence);

                                    let ltotal = l_v * l_delv * l_dj * l_delj * l_deld3;
                                    if ltotal < inference_params.min_likelihood {
                                        continue;
                                    }

                                    let (insvj, insvd) = sequence
                                        .get_insertion_vd_dj(v, delv, d, deld3, deld5, j, delj);

                                    l_total *= marginals.likelihood_nb_ins_vd(&insvj);
                                    l_total *= marginals.likelihood_nb_ins_dj(&insdj);

                                    if l_total < inference_params.min_likelihood {
                                        continue;
                                    }

                                    // add the Markov chain likelihood
                                    l_total *= marginals.likelihood_nb_ins_vd(&insvj);
                                    l_total *= marginals.likelihood_nb_ins_dj(&insvj);

                                    marginals.dirty_update(
                                        v, d, j, delv, delj, deld3, deld5, &insvd, &insdj, l_total,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
