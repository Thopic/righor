use crate::sequence::utils::{difference_as_i64, Dna};
use crate::shared::feature::*;
use crate::shared::utils::{insert_in_order, InferenceParameters};
use crate::vdj::{Event, Model, Sequence, StaticEvent};
use anyhow::Result;
use itertools::iproduct;
#[cfg(all(feature = "py_binds", feature = "py_o3"))]
use pyo3::{pyclass, pymethods};

#[derive(Default, Clone, Debug)]
#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pyclass(get_all))]
pub struct Features {
    pub v: CategoricalFeature1,
    pub delv: CategoricalFeature1g1,
    pub dj: CategoricalFeature2,
    pub delj: CategoricalFeature1g1,
    pub deld: CategoricalFeature2g1,
    // pub nb_insvd: CategoricalFeature1,
    // pub nb_insdj: CategoricalFeature1,
    pub insvd: InsertionFeature,
    pub insdj: InsertionFeature,
    pub error: ErrorPoisson,
}

impl Features {
    pub fn new(model: &Model, inference_params: &InferenceParameters) -> Result<Features> {
        Ok(Features {
            v: CategoricalFeature1::new(&model.p_v)?,
            delv: CategoricalFeature1g1::new(&model.p_del_v_given_v)?,
            dj: CategoricalFeature2::new(&model.p_dj)?,
            delj: CategoricalFeature1g1::new(&model.p_del_j_given_j)?,
            deld: CategoricalFeature2g1::new(&model.p_del_d3_del_d5)?, // dim: (d3, d5, d)
            // nb_insvd: CategoricalFeature1::new(&model.p_ins_vd)?,
            // nb_insdj: CategoricalFeature1::new(&model.p_ins_dj)?,
            insvd: InsertionFeature::new(
                &model.p_ins_vd,
                &model.first_nt_bias_ins_vd,
                &model.markov_coefficients_vd,
            )?,
            insdj: InsertionFeature::new(
                &model.p_ins_dj,
                &model.first_nt_bias_ins_dj,
                &model.markov_coefficients_dj,
            )?,
            error: ErrorPoisson::new(model.error_rate, inference_params.min_likelihood_error)?,
        })
    }

    fn log_likelihood_estimate_post_v(&self, e: &Event, m: &Model) -> f64 {
        return self.v.log_likelihood(e.v.unwrap().index) + m.max_log_likelihood_post_v;
    }

    fn log_likelihood_estimate_post_delv(&self, e: &Event, m: &Model) -> f64 {
        return self.v.log_likelihood(e.v.unwrap().index)
            + self.delv.log_likelihood((e.delv, e.v.unwrap().index))
            + self.error.log_likelihood(e.v.unwrap().nb_errors(e.delv))
            + m.max_log_likelihood_post_delv;
    }

    fn log_likelihood_estimate_post_dj(&self, e: &Event, m: &Model) -> f64 {
        let v = e.v.unwrap();
        let d = e.d.unwrap();
        let j = e.j.unwrap();
        let v_end = difference_as_i64(v.end_seq, e.delv);

        return self.v.log_likelihood(v.index)
            + self.dj.log_likelihood((d.index, j.index))
            + self.delv.log_likelihood((e.delv, v.index))
            + self.error.log_likelihood(v.nb_errors(e.delv))
        // We can already compute a fairly precise estimate of the maximum by looking at
        // the expected number of insertions
	    + m.max_log_likelihood_post_dj(d.index, (j.start_seq as i64) - v_end);
    }

    fn log_likelihood_estimate_post_delj(&self, e: &Event, m: &Model) -> f64 {
        let v = e.v.unwrap();
        let d = e.d.unwrap();
        let j = e.j.unwrap();

        let v_end = difference_as_i64(v.end_seq, e.delv);
        let j_start = (j.start_seq + e.delj) as i64;

        if v_end > j_start {
            return f64::NEG_INFINITY;
        }

        return self.v.log_likelihood(v.index)
            + self.delv.log_likelihood((e.delv, v.index))
            + self.dj.log_likelihood((d.index, j.index))
            + self.error.log_likelihood(v.nb_errors(e.delv))
            + self.delj.log_likelihood((e.delj, j.index))
            + self.error.log_likelihood(j.nb_errors(e.delj))
            + m.max_log_likelihood_post_delj(d.index, (j_start - v_end) as usize);
    }

    fn log_likelihood_estimate_post_deld(&self, e: &Event, m: &Model) -> f64 {
        let v = e.v.unwrap();
        let d = e.d.unwrap();
        let j = e.j.unwrap();

        let v_end = difference_as_i64(v.end_seq, e.delv);
        let d_start = (d.pos + e.deld5) as i64;
        let d_end = (d.pos + d.len() - e.deld3) as i64;
        let j_start = (j.start_seq + e.delj) as i64;

        if (v_end > d_start) || (d_start > d_end) || (d_end > j_start) {
            return f64::NEG_INFINITY;
        }

        return self.v.log_likelihood(v.index)
            + self.delv.log_likelihood((e.delv, v.index))
            + self.dj.log_likelihood((d.index, j.index))
            + self.error.log_likelihood(v.nb_errors(e.delv))
            + self.delj.log_likelihood((e.delj, j.index))
            + self.error.log_likelihood(j.nb_errors(e.delj))
            + self.deld.log_likelihood((e.deld3, e.deld5, d.index))
            + self.error.log_likelihood(d.nb_errors(e.deld5, e.deld3))
            + m.max_log_likelihood_post_deld(
                (d_start - v_end) as usize,
                (j_start - d_end) as usize,
            );
    }

    fn log_likelihood(&self, e: &Event, ins_vd: &Dna, ins_dj: &Dna) -> f64 {
        let v = e.v.unwrap();
        let d = e.d.unwrap();
        let j = e.j.unwrap();

        return self.v.log_likelihood(v.index)
            + self.delv.log_likelihood((e.delv, v.index))
            + self.dj.log_likelihood((d.index, j.index))
            + self.error.log_likelihood(v.nb_errors(e.delv))
            + self.delj.log_likelihood((e.delj, j.index))
            + self.error.log_likelihood(j.nb_errors(e.delj))
            + self.deld.log_likelihood((e.deld3, e.deld5, d.index))
            + self.error.log_likelihood(d.nb_errors(e.deld5, e.deld3))
            + self.insvd.log_likelihood(ins_vd)
            + self.insdj.log_likelihood(ins_dj);
    }

    // fn log_likelihood_no_error(&self, e: &Event, ins_vd: &Dna, ins_dj: &Dna) -> f64 {
    //     let v = e.v.unwrap();
    //     let d = e.d.unwrap();
    //     let j = e.j.unwrap();
    //     return self.v.log_likelihood(v.index)
    //         + self.delv.log_likelihood((e.delv, v.index))
    //         + self.dj.log_likelihood((d.index, j.index))
    //         + self.delj.log_likelihood((e.delj, j.index))
    //         + self.insvd.log_likelihood(ins_vd)
    //         + self.insdj.log_likelihood(ins_dj);
    // }

    pub fn dirty_update(&mut self, e: &Event, likelihood: f64, insvd: &Dna, insdj: &Dna) {
        let v = e.v.unwrap();
        let d = e.d.unwrap();
        let j = e.j.unwrap();
        self.v.dirty_update(v.index, likelihood);
        self.dj.dirty_update((d.index, j.index), likelihood);
        self.delv.dirty_update((e.delv, v.index), likelihood);
        self.delj.dirty_update((e.delj, j.index), likelihood);
        self.deld
            .dirty_update((e.deld3, e.deld5, d.index), likelihood);
        self.insvd.dirty_update(insvd, likelihood);
        self.insdj.dirty_update(insdj, likelihood);
        self.error.dirty_update(
            j.nb_errors(e.delj) + v.nb_errors(e.delv) + d.nb_errors(e.deld5, e.deld3),
            likelihood,
        );
    }

    pub fn infer(
        &mut self,
        sequence: &Sequence,
        m: &Model,
        ip: &InferenceParameters,
    ) -> (f64, Vec<(f64, StaticEvent)>) {
        let mut probability_generation: f64 = 0.;
        // let mut probability_generation_no_error: f64 = 0.; // TODO return this too
        let mut best_events = Vec::<(f64, StaticEvent)>::new();

        for v in sequence.v_genes.iter() {
            let e = Event {
                v: Some(v),
                ..Event::default()
            };
            if self.log_likelihood_estimate_post_v(&e, m) < ip.min_log_likelihood {
                continue;
            }
            for delv in 0..self.delv.dim().0 {
                let edelv = Event {
                    v: Some(v),
                    delv,
                    ..Event::default()
                };
                if self.log_likelihood_estimate_post_delv(&edelv, m) < ip.min_log_likelihood {
                    continue;
                }

                for (j, d) in iproduct!(sequence.j_genes.iter(), sequence.d_genes.iter()) {
                    let edj = Event {
                        v: Some(v),
                        delv,
                        d: Some(d),
                        j: Some(j),
                        ..Event::default()
                    };
                    if self.log_likelihood_estimate_post_dj(&edj, m) < ip.min_log_likelihood {
                        continue;
                    }
                    for delj in 0..self.delj.dim().0 {
                        let edelj = Event {
                            v: Some(v),
                            delv,
                            d: Some(d),
                            j: Some(j),
                            delj,
                            ..Event::default()
                        };
                        if self.log_likelihood_estimate_post_delj(&edelj, m) < ip.min_log_likelihood
                        {
                            continue;
                        }
                        for (deld5, deld3) in iproduct!(0..self.deld.dim().1, 0..self.deld.dim().0)
                        {
                            let efinal = Event {
                                v: Some(v),
                                delv,
                                d: Some(d),
                                j: Some(j),
                                delj,
                                deld5,
                                deld3,
                            };
                            if self.log_likelihood_estimate_post_deld(&efinal, m)
                                < ip.min_log_likelihood
                            {
                                continue;
                            }
                            // otherwise compute the real likelihood
                            let (insvd, insdj) = sequence.get_insertions_vd_dj(&efinal);
                            let log_likelihood = self.log_likelihood(&efinal, &insvd, &insdj);
                            if log_likelihood < ip.min_log_likelihood {
                                continue;
                            }
                            let likelihood = log_likelihood.exp2();
                            if likelihood > 1. {
                                println!("{}", log_likelihood);
                            }
                            probability_generation += likelihood;
                            self.dirty_update(&efinal, likelihood, &insvd, &insdj);

                            if ip.evaluate && ip.nb_best_events > 0 {
                                // probability_generation_no_error +=
                                //     self.likelihood_no_error(&e, &insvd, &insdj);
                                if (best_events.len() < ip.nb_best_events)
                                    || (best_events.last().unwrap().0 < likelihood)
                                {
                                    best_events = insert_in_order(
                                        best_events,
                                        (
                                            likelihood,
                                            efinal.to_static(insvd.clone(), insdj.clone()).unwrap(),
                                        ),
                                    );
                                    best_events.truncate(ip.nb_best_events);
                                }
                            }
                        }
                    }
                }
            }
        }
        (probability_generation, best_events)
    }

    pub fn cleanup(&self) -> Result<Features> {
        // Compute the new marginals for the next round
        Ok(Features {
            v: self.v.cleanup()?,
            dj: self.dj.cleanup()?,
            delv: self.delv.cleanup()?,
            delj: self.delj.cleanup()?,
            deld: self.deld.cleanup()?,
            insvd: self.insvd.cleanup()?,
            insdj: self.insdj.cleanup()?,
            error: self.error.cleanup()?,
        })
    }
}

#[cfg(not(all(feature = "py_binds", feature = "py_o3")))]
impl Features {
    pub fn average(features: Vec<Features>) -> Result<Features> {
        Ok(Features {
            v: CategoricalFeature1::average(features.iter().map(|a| a.v.clone()))?,
            delv: CategoricalFeature1g1::average(features.iter().map(|a| a.delv.clone()))?,
            dj: CategoricalFeature2::average(features.iter().map(|a| a.dj.clone()))?,
            delj: CategoricalFeature1g1::average(features.iter().map(|a| a.delj.clone()))?,
            deld: CategoricalFeature2g1::average(features.iter().map(|a| a.deld.clone()))?,
            insvd: InsertionFeature::average(features.iter().map(|a| a.insvd.clone()))?,
            insdj: InsertionFeature::average(features.iter().map(|a| a.insdj.clone()))?,
            error: ErrorPoisson::average(features.iter().map(|a| a.error.clone()))?,
        })
    }
}

#[cfg(all(feature = "py_binds", feature = "py_o3"))]
#[pymethods]
impl Features {
    #[staticmethod]
    pub fn average(features: Vec<Features>) -> Result<Features> {
        Ok(Features {
            v: CategoricalFeature1::average(features.iter().map(|a| a.v.clone()))?,
            delv: CategoricalFeature1g1::average(features.iter().map(|a| a.delv.clone()))?,
            dj: CategoricalFeature2::average(features.iter().map(|a| a.dj.clone()))?,
            delj: CategoricalFeature1g1::average(features.iter().map(|a| a.delj.clone()))?,
            deld: CategoricalFeature2g1::average(features.iter().map(|a| a.deld.clone()))?,
            insvd: InsertionFeature::average(features.iter().map(|a| a.insvd.clone()))?,
            insdj: InsertionFeature::average(features.iter().map(|a| a.insdj.clone()))?,
            error: ErrorPoisson::average(features.iter().map(|a| a.error.clone()))?,
        })
    }
}
