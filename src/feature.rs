// A sequence is a collection of feature, each feature has some probability
// When we iterate on the model we update the probability of each feature

use crate::model::ModelVDJ;
use crate::sequence::SequenceVDJ;
use crate::utils::{likelihood_markov, update_markov_probas, Normalize};
use crate::utils_sequences::Dna;
use ndarray::{Array1, Array2, Array3, Axis};
use std::error::Error;

pub struct InferenceParams {
    min_likelihood: f64,
    nb_rounds_em: usize,
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

#[derive(Default, Clone, Debug)]
pub struct MarginalsVDJ {
    // the marginals used during the inference
    marginals: FeaturesVDJ,

    // this variable store the evolving marginals as they
    // are iterated upon (in particular it's not expected to
    // be normalized)
    dirty_marginals: FeaturesVDJ,
}

impl FeaturesVDJ {
    fn normalize(&mut self) -> Result<FeaturesVDJ, Box<dyn Error>> {
        // Normalize the arrays
        // Because of the way the original code is set up
        // some distributions are identically 0 (p(delv|v=v1) if
        // v1 has probability 0 for example)
        // We modify these probability to make them uniform
        // (and avoid issues like log(0))
        Ok(FeaturesVDJ {
            v: self.v.normalize_distribution(None)?,
            dj: self
                .dj
                .normalize_distribution(Some(Axis(0)))?
                .normalize_distribution(Some(Axis(1)))?,
            delv: self.delv.normalize_distribution(Some(Axis(0)))?,
            delj: self.delj.normalize_distribution(Some(Axis(0)))?,
            deld: self
                .deld
                .normalize_distribution(Some(Axis(0)))?
                .normalize_distribution(Some(Axis(1)))?,
            insvd: self.insvd.normalize_distribution(None)?,
            insdj: self.insdj.normalize_distribution(None)?,
            first_nt_bias_vd: self.first_nt_bias_vd.normalize_distribution(None)?,
            first_nt_bias_dj: self.first_nt_bias_dj.normalize_distribution(None)?,
            markov_coefficients_vd: self
                .markov_coefficients_vd
                .normalize_distribution(Some(Axis(1)))?,
            markov_coefficients_dj: self
                .markov_coefficients_dj
                .normalize_distribution(Some(Axis(1)))?,
        })
    }
}

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
        self.marginals.delv[[dv, vi]]
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
        nb_insvd: usize,
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

    fn expectation_step(&mut self) {
        // Normalize the dirty marginals and move them to the
        // current marginals
        self.marginals = self.dirty_marginals.normalize();
    }

    fn maximization_step(
        &mut self,
        sequences: &Vec<SequenceVDJ>,
        inference_params: &InferenceParams,
    ) {
        for s in sequences {
            self.update_marginals(s, inference_params);
        }
    }

    fn expectation_maximization(
        &mut self,
        sequences: &Vec<SequenceVDJ>,
        inference_params: &InferenceParams,
    ) {
        for _ in inference_params.number_rounds_em {
            self.maximization_step(sequences, inference_params);
            self.expectation_step();
        }
    }

    fn update_marginals(&mut self, sequence: &SequenceVDJ, inference_params: &InferenceParams) {
        for v in sequence.v_genes {
            let l_v = self.likelihood_v(v.index);
            for delv in 0..self.marginals.p_delv.dim()[[0]] {
                let l_delv = self.likelihood_delv(delv, v.index, &sequence);
                for j in sequence.j_genes {
                    for d in sequence.d_genes {
                        let l_dj = self.likelihood_dj(d.index, j.index);
                        for delj in 0..self.marginals.p_delj.dim()[[0]] {
                            let l_delj = self.likelihood_delj(delj, j.index, &sequence);

                            if l_delj * l_dj * l_delv * l_v < inference_params.min_likelihood {
                                continue;
                            }

                            for deld3 in 0..self.marginals.p_deld3.dim()[[0]] {
                                for deld5 in 0..self.marginals.p_deld5.dim()[[1]] {
                                    let (l_deld3, l_deld5) =
                                        self.likelihood_deld(deld3, deld5, d, &sequence);

                                    let l_total = l_v * l_delv * l_dj * l_delj * l_deld3;
                                    if l_total < inference_params.min_likelihood {
                                        continue;
                                    }

                                    let (insvd, insdj) = sequence
                                        .get_insertion_vd_dj(v, delv, d, deld3, deld5, j, delj);

                                    l_total *= self.likelihood_nb_ins_vd(&insvd);
                                    l_total *= self.likelihood_nb_ins_dj(&insdj);

                                    if l_total < inference_params.min_likelihood {
                                        continue;
                                    }

                                    // add the Markov chain likelihood
                                    l_total *= self.likelihood_nb_ins_vd(&insvd);
                                    l_total *= self.likelihood_nb_ins_dj(&insdj);

                                    self.dirty_update(
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
