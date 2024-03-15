use crate::sequence::AlignmentParameters;
use crate::sequence::Dna;
use crate::shared::parser::{parse_file, parse_str, ParserMarginals, ParserParams};
use crate::shared::utils::{
    calc_steady_state_dist, sorted_and_complete, sorted_and_complete_0start, Gene, RecordModel,
};
use crate::shared::InferenceParameters;
use crate::vdj::{
    inference::{Features, InfEvent, ResultInference},
    Model as ModelVDJ, Sequence,
};
use crate::vj::StaticEvent;
use anyhow::{anyhow, Result};
use ndarray::{array, Array1, Array2, Array3, Axis};

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{fs::read_to_string, fs::File, path::Path};

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct Generator {
    model: Model,
    rng: SmallRng,
}

impl Generator {
    pub fn new(
        model: Model,
        seed: Option<u64>,
        available_v: Option<Vec<Gene>>,
        available_j: Option<Vec<Gene>>,
    ) -> Result<Generator> {
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };

        // create an internal model in case we need to restrict the V/J genes.
        let mut internal_model = model.clone();

        if !available_v.is_none() {
            internal_model = internal_model.filter_vs(available_v.unwrap())?;
        }
        if !available_j.is_none() {
            internal_model = internal_model.filter_js(available_j.unwrap())?;
        }
        Ok(Generator {
            model: internal_model,
            rng,
        })
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl Generator {
    pub fn generate(&mut self, functional: bool) -> GenerationResult {
        self.model.generate(functional, &mut self.rng)
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct GenerationResult {
    pub cdr3_nt: String,
    pub cdr3_aa: Option<String>,
    pub full_seq: String,
    pub v_gene: String,
    pub j_gene: String,
    pub recombination_event: StaticEvent,
}

// A VJ model is in practice a simplified VDJ model (without insDJ / D / delD3 / delD5)
// So I use a VDJ model as the inner model, with a different parameter set.
#[derive(Default, Clone, Debug)]
pub struct Model {
    // The actual/real underlying model
    pub inner: ModelVDJ,

    // Sequence information
    pub seg_vs: Vec<Gene>,
    pub seg_js: Vec<Gene>,

    // V/J nucleotides sequences trimmed at the CDR3 region (include F/W/C residues) with
    // the maximum number of reverse palindromic insertions appended.
    pub seg_vs_sanitized: Vec<Dna>, // match genomic_data.cutV_genomic_CDR3_seqs
    pub seg_js_sanitized: Vec<Dna>,

    // Probabilities of the different events
    pub p_v: Array1<f64>,
    pub p_j_given_v: Array2<f64>,
    pub p_ins_vj: Array1<f64>,
    pub p_del_v_given_v: Array2<f64>,
    pub p_del_j_given_j: Array2<f64>,
    pub markov_coefficients_vj: Array2<f64>,
    pub range_del_v: (i64, i64),
    pub range_del_j: (i64, i64),
    pub error_rate: f64,

    // Not directly useful for the model but useful for integration with other soft
    // TODO: transform these in "getter" in the python bindings, they don't need to be precomputed
    pub first_nt_bias_ins_vj: Array1<f64>,
    pub thymic_q: f64,
}

impl Model {
    pub fn load_from_name(
        species: &str,
        chain: &str,
        id: Option<String>,
        model_dir: &Path,
    ) -> Result<Model> {
        let content = read_to_string(model_dir.join("models.json"))?;
        let records: Vec<RecordModel> = serde_json::from_str(&content)?;

        for record in records {
            if record.species.contains(&species.to_string().to_lowercase())
                && record.chain.contains(&chain.to_string().to_lowercase())
                && id.as_ref().map_or(true, |i| &record.id == i)
            {
                return Self::load_from_files(
                    &model_dir.join(Path::new(&record.filename_params)),
                    &model_dir.join(Path::new(&record.filename_marginals)),
                    &model_dir.join(Path::new(&record.filename_v_gene_cdr3_anchors)),
                    &model_dir.join(Path::new(&record.filename_j_gene_cdr3_anchors)),
                );
            }
        }

        if id.is_none() {
            Err(anyhow!(
                "The given species ({}) / chain ({}) don't match any model",
                species,
                chain
            ))
        } else {
            Err(anyhow!(
                "The given species ({}) / chain ({}) / id ({}) don't match any model",
                species,
                chain,
                id.unwrap()
            ))
        }
    }

    pub fn load_from_files(
        path_params: &Path,
        path_marginals: &Path,
        path_anchor_vgene: &Path,
        path_anchor_jgene: &Path,
    ) -> Result<Model> {
        let pm: ParserMarginals = ParserMarginals::parse(parse_file(path_marginals)?)?;
        let mut pp: ParserParams = ParserParams::parse(parse_file(path_params)?)?;

        let rdr_v =
            File::open(path_anchor_vgene).map_err(|_e| anyhow!("Error opening the anchor file"))?;
        let rdr_j =
            File::open(path_anchor_jgene).map_err(|_e| anyhow!("Error opening the anchor file"))?;

        pp.add_anchors_gene(rdr_v, "v_choice")?;
        pp.add_anchors_gene(rdr_j, "j_choice")?;
        Self::load_model(&pp, &pm)
    }

    pub fn load_from_str(
        params: &str,
        marginals: &str,
        anchor_vgene: &str,
        anchor_jgene: &str,
    ) -> Result<Model> {
        let pm: ParserMarginals = ParserMarginals::parse(parse_str(marginals)?)?;
        let mut pp: ParserParams = ParserParams::parse(parse_str(params)?)?;

        let rdr_v = anchor_vgene.as_bytes();
        let rdr_j = anchor_jgene.as_bytes();

        pp.add_anchors_gene(rdr_v, "v_choice")?;
        pp.add_anchors_gene(rdr_j, "j_choice")?;
        Self::load_model(&pp, &pm)
    }

    pub fn load_model(pp: &ParserParams, pm: &ParserMarginals) -> Result<Model> {
        let mut model: Model = Model {
            seg_vs: pp
                .params
                .get("v_choice")
                .ok_or(anyhow!("Error with unwrapping the Params data"))?
                .clone()
                .to_genes()?,
            seg_js: pp
                .params
                .get("j_choice")
                .ok_or(anyhow!("Error with unwrapping the Params data"))?
                .clone()
                .to_genes()?,
            ..Default::default()
        };

        let arrdelv = pp
            .params
            .get("v_3_del")
            .ok_or(anyhow!("Invalid v_3_del"))?
            .clone()
            .to_numbers()?;

        model.range_del_v = (
            *arrdelv.iter().min().ok_or(anyhow!("Empty v_3_del"))?,
            *arrdelv.iter().max().ok_or(anyhow!("Empty v_3_del"))?,
        );
        let arrdelj = pp
            .params
            .get("j_5_del")
            .ok_or(anyhow!("Invalid j_5_del"))?
            .clone()
            .to_numbers()?;
        model.range_del_j = (
            *arrdelj.iter().min().ok_or(anyhow!("Empty j_5_del"))?,
            *arrdelj.iter().max().ok_or(anyhow!("Empty j_5_del"))?,
        );

        if !(sorted_and_complete(arrdelv)
            & sorted_and_complete(arrdelj)
            & sorted_and_complete_0start(
                pp.params
                    .get("vj_ins")
                    .ok_or(anyhow!("Invalid vj_ins"))?
                    .clone()
                    .to_numbers()?,
            ))
        {
            return Err(anyhow!(
                "The number of insertion or deletion in the model parameters should\
			be sorted and should not contain missing value. E.g.:\n\
			%0;0\n\
			%12;1\n\
			or: \n\
			%0;1\n\
			%1;0\n\
			will both result in this error."
            ));
        }

        // Set the different probabilities for the model
        model.p_v = pm
            .marginals
            .get("v_choice")
            .unwrap()
            .probabilities
            .clone()
            .into_dimensionality()
            .unwrap();
        model.p_j_given_v = pm
            .marginals
            .get("j_choice")
            .unwrap()
            .probabilities
            .clone()
            .into_dimensionality()
            .unwrap()
            .t()
            .to_owned();
        model.p_del_v_given_v = pm
            .marginals
            .get("v_3_del")
            .unwrap()
            .probabilities
            .clone()
            .into_dimensionality()
            .unwrap()
            .t()
            .to_owned();
        model.p_del_j_given_j = pm
            .marginals
            .get("j_5_del")
            .unwrap()
            .probabilities
            .clone()
            .into_dimensionality()
            .unwrap()
            .t()
            .to_owned();

        model.p_ins_vj = pm
            .marginals
            .get("vj_ins")
            .unwrap()
            .probabilities
            .clone()
            .into_dimensionality()
            .unwrap();

        // Markov coefficients
        model.markov_coefficients_vj = pm
            .marginals
            .get("vj_dinucl")
            .unwrap()
            .probabilities
            .clone()
            .into_shape((4, 4))
            .map_err(|_e| anyhow!("Wrong size for vj_dinucl"))?;

        // TODO: Need to deal with potential first nt bias
        model.first_nt_bias_ins_vj =
            Array1::from_vec(calc_steady_state_dist(&model.markov_coefficients_vj)?);

        model.error_rate = pp.error_rate;
        model.thymic_q = 9.41; // TODO: deal with this

        model.initialize()?;
        Ok(model)
    }

    pub fn initialize(&mut self) -> Result<()> {
        self.load_inner_vdj()?;
        Ok(())
    }

    fn load_inner_vdj(&mut self) -> Result<()> {
        // create an empty d gene
        let d_gene = Gene {
            name: "Empty_D_gene".to_string(),
            seq: Dna::new(),
            seq_with_pal: Some(Dna::new()),
            ..Default::default()
        };

        self.inner = ModelVDJ {
            seg_vs: self.seg_vs.clone(),
            seg_js: self.seg_js.clone(),
            seg_ds: vec![d_gene],
            p_ins_vd: self.p_ins_vj.clone(),
            p_ins_dj: array![1.], // only one option: no insertion
            p_del_v_given_v: self.p_del_v_given_v.clone(),
            p_del_j_given_j: self.p_del_j_given_j.clone(),
            p_del_d5_del_d3: array![[[1.]]], // one option, no deletion, empty D gene.
            markov_coefficients_vd: self.markov_coefficients_vj.clone(),
            // just need to give it some value
            markov_coefficients_dj: self.markov_coefficients_vj.clone(),
            range_del_v: self.range_del_v,
            range_del_j: self.range_del_j,
            range_del_d3: (0, 0),
            range_del_d5: (0, 0),
            error_rate: self.error_rate,
            ..Default::default()
        };

        let vdim = self.p_v.dim();
        let jdim = self.p_j_given_v.dim().0;

        let mut p_vdj = Array3::<f64>::zeros((vdim, 1, jdim));
        for vv in 0..vdim {
            for jj in 0..jdim {
                p_vdj[[vv, 0, jj]] = self.p_j_given_v[[jj, vv]] * self.p_v[[vv]];
            }
        }

        self.inner.set_p_vdj(&p_vdj)?;
        self.inner.initialize()?;

        Ok(())
    }

    fn update_outer_model(&mut self) -> Result<()> {
        self.seg_vs = self.inner.seg_vs.clone();
        self.seg_js = self.inner.seg_js.clone();
        self.seg_vs_sanitized = self.inner.seg_vs_sanitized.clone();
        self.seg_js_sanitized = self.inner.seg_js_sanitized.clone();
        self.p_ins_vj = self.inner.p_ins_vd.clone();
        self.p_del_v_given_v = self.inner.p_del_v_given_v.clone();
        self.p_del_j_given_j = self.inner.p_del_j_given_j.clone();
        self.p_v = self.inner.p_v.clone();
        self.p_j_given_v = self.inner.p_j_given_v.clone();
        self.p_ins_vj = self.inner.p_ins_vd.clone();
        self.markov_coefficients_vj = self.inner.markov_coefficients_vd.clone();
        self.first_nt_bias_ins_vj = self.inner.first_nt_bias_ins_vd.clone();
        self.range_del_j = self.inner.range_del_j;
        self.range_del_v = self.inner.range_del_v;
        self.error_rate = self.inner.error_rate;
        self.initialize()?;
        Ok(())
    }

    // pub fn generate_cdr3_no_error<R: Rng>(
    //     &mut self,
    //     functional: bool,
    //     rng: &mut R,
    // ) -> (Dna, Option<AminoAcid>, StaticEvent) {
    //     let (seq, aa, se) = self.inner.generate_cdr3no_error(functional, rng);
    //     (
    //         seq,
    //         aa,
    //         StaticEvent {
    //             v_index: se.v_index,
    //             v_start_gene: se.v_start_gene,
    //             delv: se.delv,
    //             j_index: se.j_index,
    //             j_start_seq: se.j_start_seq,
    //             delj: se.delj,
    //             insvj: se.insvd,
    //         },
    //     )
    // }

    /// Return (cdr3_nt, cdr3_aa, full_sequence, event, vname, jname)
    pub fn generate<R: Rng>(&mut self, functional: bool, rng: &mut R) -> GenerationResult {
        let gen_result = self.inner.generate(functional, rng);
        let se = gen_result.recombination_event;
        GenerationResult {
            cdr3_nt: gen_result.cdr3_nt,
            cdr3_aa: gen_result.cdr3_aa,
            full_seq: gen_result.full_seq,
            v_gene: gen_result.v_gene,
            j_gene: gen_result.j_gene,
            recombination_event: StaticEvent {
                v_index: se.v_index,
                v_start_gene: se.v_start_gene,
                delv: se.delv,
                j_index: se.j_index,
                j_start_seq: se.j_start_seq,
                delj: se.delj,
                insvj: se.insvd,
            },
        }
    }

    pub fn filter_vs(&self, vs: Vec<Gene>) -> Result<Model> {
        let mut m = Model {
            inner: self.inner.filter_vs(vs)?,
            ..Default::default()
        };
        m.update_outer_model()?;
        m.initialize()?;
        Ok(m)
    }

    pub fn filter_js(&self, js: Vec<Gene>) -> Result<Model> {
        let mut m = Model {
            inner: self.inner.filter_js(js)?,
            ..Default::default()
        };
        m.update_outer_model()?;
        m.initialize()?;
        Ok(m)
    }

    /// Return an uniform model (for initializing the inference)
    pub fn uniform(&self) -> Result<Model> {
        let mut m = Model {
            inner: self.inner.uniform()?,
            ..Default::default()
        };
        m.update_outer_model()?;
        m.initialize()?;
        Ok(m)
    }

    pub fn evaluate(
        &self,
        sequence: &Sequence,
        inference_params: &InferenceParameters,
    ) -> Result<ResultInference> {
        self.inner.evaluate(sequence, inference_params)
    }

    pub fn infer(
        &mut self,
        sequences: &Vec<Sequence>,
        inference_params: &InferenceParameters,
    ) -> Result<()> {
        self.inner.infer(sequences, inference_params)?;
        self.update_outer_model()?;
        Ok(())
    }

    pub fn get_v_gene(&self, event: &InfEvent) -> String {
        self.seg_vs[event.v_index].name.clone()
    }

    pub fn get_j_gene(&self, event: &InfEvent) -> String {
        self.seg_js[event.j_index].name.clone()
    }

    pub fn get_p_j(&self) -> Array1<f64> {
        (self.get_p_vj()).sum_axis(Axis(1))
    }

    pub fn get_p_vj(&self) -> Array2<f64> {
        self.p_j_given_v.clone() * self.p_v.clone()
    }

    pub fn set_p_vj(&mut self, arr: &Array2<f64>) -> Result<()> {
        self.p_v = Array1::<f64>::zeros(arr.shape()[0]);
        self.p_j_given_v = Array2::<f64>::zeros((arr.shape()[1], arr.shape()[0]));
        for ii in 0..arr.shape()[0] {
            for jj in 0..arr.shape()[1] {
                self.p_v[ii] += arr[[ii, jj]];
            }
        }
        for ii in 0..arr.shape()[0] {
            for jj in 0..arr.shape()[1] {
                self.p_j_given_v[[jj, ii]] = arr[[ii, jj]] / self.p_v[ii]
            }
        }
        self.initialize()?;
        Ok(())
    }

    pub fn align_sequence(
        &self,
        dna_seq: Dna,
        align_params: &AlignmentParameters,
    ) -> Result<Sequence> {
        self.inner.align_sequence(dna_seq, align_params)
    }

    pub fn recreate_full_sequence(
        &self,
        dna: &Dna,
        v_index: usize,
        j_index: usize,
    ) -> (Dna, String, String, usize) {
        self.inner.recreate_full_sequence(dna, v_index, j_index)
    }

    pub fn from_features(&self, feature: &Features) -> Result<Model> {
        let mut m = self.clone();
        m.inner.update(feature)?;
        m.update_outer_model()?;
        m.initialize()?;
        Ok(m)
    }
}

//     fn initialize_generative_model(&mut self) -> Result<()> {
//         self.gen.d_v = DiscreteDistribution::new(self.p_v.to_vec())?;
//         self.gen.d_ins_vj = DiscreteDistribution::new(self.p_ins_vj.to_vec())?;

//         self.gen.d_j_given_v = Vec::new();
//         for row in self.p_j_given_v.axis_iter(Axis(1)) {
//             self.gen
//                 .d_j_given_v
//                 .push(DiscreteDistribution::new(row.to_vec())?);
//         }

//         self.gen.d_del_v_given_v = Vec::new();
//         for row in self.p_del_v_given_v.axis_iter(Axis(1)) {
//             self.gen
//                 .d_del_v_given_v
//                 .push(DiscreteDistribution::new(row.to_vec())?);
//         }
//         self.gen.d_del_j_given_j = Vec::new();
//         for row in self.p_del_j_given_j.axis_iter(Axis(1)) {
//             self.gen
//                 .d_del_j_given_j
//                 .push(DiscreteDistribution::new(row.to_vec())?);
//         }

//         self.gen.markov_vj = MarkovDNA::new(self.markov_coefficients_vj.t().to_owned(), None)?;
//         Ok(())
//     }

//     pub fn generate_cdr3_no_error<R: Rng>(
//         &mut self,
//         functional: bool,
//         rng: &mut R,
//     ) -> (Dna, Option<AminoAcid>, StaticEvent) {
//         // loop until we find a valid sequence (if generating functional alone)
//         loop {
//             let mut event = StaticEvent {
//                 ..Default::default()
//             };

//             event.v_index = self.gen.d_v.generate(rng);
//             event.j_index = self.gen.d_j_given_v[v_index].generate(rng);

//             let seq_v_cdr3: &Dna = &self.seg_vs_sanitized[v_index];
//             let seq_j_cdr3: &Dna = &self.seg_js_sanitized[j_index];

//             let seq_v: &Dna = self.seg_vs[event.v_index].seq_with_pal.as_ref().unwrap();
//             let seq_j: &Dna = self.seg_js[event.j_index].seq_with_pal.as_ref().unwrap();

//             event.delv = self.gen.d_del_v_given_v[event.v_index].generate(rng);
//             event.delj = self.gen.d_del_j_given_j[event.j_index].generate(rng);

//             let ins_vj: usize = self.gen.d_ins_vj.generate(rng);

//             let out_of_frame =
//                 (seq_v_cdr3.len() - event.del_v + seq_j_cdr3.len() - event.del_j + ins_vj) % 3 != 0;
//             if functional & out_of_frame {
//                 continue;
//             }

//             // look at the last nucleotide of V (for the Markov chain)
//             let end_v = seq_v.seq[seq_v.len() - event.delv - 1];
//             event.insvj = self.gen.markov_vj.generate(ins_vj, end_v, rng);

//             event.v_start_gene = 0;
//             event.j_start_seq = seq_v.len() - event.delv + ins_vj - event.delj;

//             // create the complete sequence
//             let seq = event.to_cdr3();

//             // translate
//             let seq_aa: Option<AminoAcid> = seq.translate().ok();

//             match seq_aa {
//                 Some(saa) => {
//                     // check for stop codon
//                     if functional & saa.seq.contains(&b'*') {
//                         continue;
//                     }

//                     // check for conserved extremities (cysteine)
//                     if functional & (saa.seq[0] != b'C') {
//                         continue;
//                     }
//                     return (seq, Some(saa), event);
//                 }
//                 None => {
//                     if functional {
//                         continue;
//                     }
//                     return (seq, None, event);
//                 }
//             }
//         }
//     }

//     /// Return (cdr3_nt, cdr3_aa, full_sequence, event, vname, jname)
//     pub fn generate<R: Rng>(
//         &mut self,
//         functional: bool,
//         rng: &mut R,
//     ) -> (Dna, Option<AminoAcid>, Dna, StaticEvent, String, String) {
//         let (cdr3_nt, _, event) = self.generate_cdr3_no_error(functional, rng);
//         let (mut full_seq, vname, jname, start_cdr3) =
//             self.recreate_full_sequence(&cdr3_nt, event.v_index, event.j_index);
//         // add potential sequencing error
//         add_errors(&mut full_seq, self.error_rate, rng);
//         let cdr3_err = full_seq.extract_subsequence(start_cdr3, start_cdr3 + cdr3_nt.len());
//         let seq_aa: Option<AminoAcid> = cdr3_err.translate().ok();
//         (cdr3_err, seq_aa, full_seq, event, vname, jname)
//     }

//     pub fn uniform(&self) -> Result<Model> {
//         let mut m = Model {
//             seg_vs: self.seg_vs.clone(),
//             seg_js: self.seg_js.clone(),
//             seg_ds: self.seg_ds.clone(),

//             range_del_v: self.range_del_v,
//             range_del_j: self.range_del_j,
//             p_v: Array1::<f64>::ones(self.p_v.dim()),
//             p_j_given_v: Array2::<f64>::ones(self.p_j_given_v.dim()),
//             p_ins_vj: Array1::<f64>::ones(self.p_ins_vj.dim()),
//             p_del_v_given_v: Array2::<f64>::ones(self.p_del_v_given_v.dim()),
//             p_del_j_given_j: Array2::<f64>::ones(self.p_del_j_given_j.dim()),
//             markov_coefficients_vj: Array2::<f64>::ones(self.markov_coefficients_vj.dim()),
//             first_nt_bias_ins_vj: Array1::<f64>::ones(self.first_nt_bias_ins_vj.dim()),
//             error_rate: 0.1, // TODO: too ad-hoc
//             ..Default::default()
//         };
//         m.initialize()?;
//         Ok(m)
//     }

//     pub fn evaluate(
//         &self,
//         sequence: &Sequence,
//         inference_params: &InferenceParameters,
//     ) -> Result<ResultInference> {
//         let mut feature = Features::new(self)?;
//         let mut result = feature.infer(sequence, inference_params)?;
//         result.fill_event(self, sequence)?;
//         result.features = Some(feature.clone());
//         Ok(result)
//     }

//     pub fn infer(
//         &mut self,
//         sequences: &Vec<Sequence>,
//         inference_params: &InferenceParameters,
//     ) -> Result<()> {
//         let mut ip = inference_params.clone();
//         ip.infer = true;
//         let features = sequences
//             .par_iter()
//             .map(|sequence| {
//                 let mut feature = Features::new(self)?;
//                 let _ = feature.infer(sequence, &ip)?;
//                 Ok(feature)
//             })
//             .collect::<Result<Vec<_>>>()?;

//         let avg_features = Features::average(features)?;
//         self.update(&avg_features)?;
//         Ok(())
//     }

//     pub fn get_v_gene(&self, event: &InfEvent) -> String {
//         self.seg_vs[event.v_index].name.clone()
//     }

//     pub fn get_j_gene(&self, event: &InfEvent) -> String {
//         self.seg_js[event.j_index].name.clone()
//     }
// }

// #[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
// impl Model {
//     pub fn infer_features(
//         &self,
//         sequence: &Sequence,
//         inference_params: &InferenceParameters,
//     ) -> Result<Features> {
//         let mut feature = Features::new(self, inference_params)?;
//         let (ltotal, _) = feature.infer(sequence, inference_params, 0);
//         if ltotal == 0.0f64 {
//             return Ok(feature); // return 0s.
//         }
//         // Otherwise normalize
//         feature = feature.cleanup()?;
//         Ok(feature)
//     }
//     pub fn most_likely_recombinations(
//         &self,
//         sequence: &Sequence,
//         nb_scenarios: usize,
//         inference_params: &InferenceParameters,
//     ) -> Result<Vec<(f64, StaticEvent)>> {
//         let mut feature = Features::new(self, inference_params)?;
//         let (_, res) = feature.infer(sequence, inference_params, nb_scenarios);
//         Ok(res)
//     }
//     pub fn pgen(&self, sequence: &Sequence, inference_params: &InferenceParameters) -> Result<f64> {
//         let mut feature = Features::new(self, inference_params)?;
//         let (pg, _) = feature.infer(sequence, inference_params, 0);
//         Ok(pg)
//     }

//     pub fn align_sequence(
//         &self,
//         dna_seq: Dna,
//         align_params: &AlignmentParameters,
//     ) -> Result<Sequence> {
//         let seq = Sequence {
//             sequence: dna_seq.clone(),
//             v_genes: align_all_vgenes(&dna_seq, self, align_params),
//             j_genes: align_all_jgenes(&dna_seq, self, align_params),
//             d_genes: Vec::new(),
//         };
//         Ok(seq)
//     }

//     pub fn recreate_full_sequence(
//         &self,
//         dna: &Dna,
//         v_index: usize,
//         j_index: usize,
//     ) -> (Dna, String, String) {
//         // Re-create the full sequence of the variable region (with complete V/J gene, not just the CDR3)
//         let mut seq: Dna = Dna::new();
//         let vgene = self.seg_vs[v_index].clone();
//         let jgene = self.seg_js[j_index].clone();
//         seq.extend(&vgene.seq.extract_subsequence(0, vgene.cdr3_pos.unwrap()));
//         seq.extend(dna);
//         seq.extend(
//             &jgene
//                 .seq
//                 .extract_subsequence(jgene.cdr3_pos.unwrap() + 3, jgene.seq.len()),
//         );
//         (seq, vgene.name, jgene.name)
//     }

//     pub fn update(&mut self, feature: &Features) {
//         self.p_v = feature.v.log_probas.map(|x| x.exp2());
//         self.p_del_v_given_v = feature.delv.log_probas.map(|x| x.exp2());
//         self.p_j_given_v = feature.j.log_probas.map(|x| x.exp2());
//         self.p_del_j_given_j = feature.delj.log_probas.map(|x| x.exp2());
//         self.p_ins_vj = feature.nb_insvj.log_probas.map(|x| x.exp2());
//         (
//             self.p_ins_vj,
//             self.first_nt_bias_ins_vj,
//             self.markov_coefficients_vj,
//         ) = feature.insvj.get_parameters();
//         self.error_rate = feature.error.error_rate;
//     }

//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[staticmethod]
//     #[pyo3(name = "load_model")]
//     pub fn py_load_model(
//         path_params: &str,
//         path_marginals: &str,
//         path_anchor_vgene: &str,
//         path_anchor_jgene: &str,
//     ) -> Result<Model> {
//         Model::load_model(
//             Path::new(path_params),
//             Path::new(path_marginals),
//             Path::new(path_anchor_vgene),
//             Path::new(path_anchor_jgene),
//         )
//     }

//     // getter & setter for the numpy/ndarray arrays, no easy way to make them automatically
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[getter]
//     fn get_p_v(&self, py: Python) -> Py<PyArray1<f64>> {
//         self.p_v.to_owned().into_pyarray(py).to_owned()
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[setter]
//     fn set_p_v(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
//         self.p_v = value.as_ref(py).to_owned_array();
//         Ok(())
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[getter]
//     fn get_p_j_given_v(&self, py: Python) -> Py<PyArray2<f64>> {
//         self.p_j_given_v.to_owned().into_pyarray(py).to_owned()
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[setter]
//     fn set_p_j_given_v(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
//         self.p_j_given_v = value.as_ref(py).to_owned_array();
//         Ok(())
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[getter]
//     fn get_p_ins_vj(&self, py: Python) -> Py<PyArray1<f64>> {
//         self.p_ins_vj.to_owned().into_pyarray(py).to_owned()
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[setter]
//     fn set_p_ins_vj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
//         self.p_ins_vj = value.as_ref(py).to_owned_array();
//         Ok(())
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[getter]
//     fn get_p_del_v_given_v(&self, py: Python) -> Py<PyArray2<f64>> {
//         self.p_del_v_given_v.to_owned().into_pyarray(py).to_owned()
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[setter]
//     fn set_p_del_v_given_v(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
//         self.p_del_v_given_v = value.as_ref(py).to_owned_array();
//         Ok(())
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[getter]
//     fn get_p_del_j_given_j(&self, py: Python) -> Py<PyArray2<f64>> {
//         self.p_del_j_given_j.to_owned().into_pyarray(py).to_owned()
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[setter]
//     fn set_p_del_j_given_j(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
//         self.p_del_j_given_j = value.as_ref(py).to_owned_array();
//         Ok(())
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[getter]
//     fn get_markov_coefficients_vj(&self, py: Python) -> Py<PyArray2<f64>> {
//         self.markov_coefficients_vj
//             .to_owned()
//             .into_pyarray(py)
//             .to_owned()
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[setter]
//     fn set_markov_coefficients_vj(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
//         self.markov_coefficients_vj = value.as_ref(py).to_owned_array();
//         Ok(())
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[getter]
//     fn get_first_nt_bias_ins_vj(&self, py: Python) -> Py<PyArray1<f64>> {
//         self.first_nt_bias_ins_vj
//             .to_owned()
//             .into_pyarray(py)
//             .to_owned()
//     }
//     #[cfg(all(feature = "py_binds", feature = "pyo3"))]
//     #[setter]
//     fn set_first_nt_bias_ins_vj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
//         self.first_nt_bias_ins_vj = value.as_ref(py).to_owned_array();
//         Ok(())
//     }
// }
