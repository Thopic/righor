use crate::sequence::AlignmentParameters;
use crate::sequence::Dna;
use crate::shared::parser::{
    parse_file, parse_str, EventType, Marginal, ParserMarginals, ParserParams,
};
use crate::shared::utils::{
    calc_steady_state_dist, sorted_and_complete, sorted_and_complete_0start, Gene, ModelGen,
    RecordModel,
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
use std::{fs::read_to_string, fs::File, io::Write, path::Path};

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
    pub fn generate_without_errors(&mut self, functional: bool) -> GenerationResult {
        self.model
            .generate_without_errors(functional, &mut self.rng)
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

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl GenerationResult {
    fn __repr__(&self) -> String {
        format!(
            "GenerationResult(\n\
		 CDR3 (nucletides): {},\n\
		 CDR3 (amino-acids): {},\n\
		 Full sequence (nucleotides): {}...,\n\
		 V gene: {},\n\
		 J gene: {})
		 ",
            self.cdr3_nt,
            self.cdr3_aa.clone().unwrap_or("Out-of-frame".to_string()),
            &self.full_seq[0..30],
            self.v_gene,
            self.j_gene
        )
    }
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

    /// Save the model in a given directory (write 4 files)
    pub fn save_model(&self, directory: &Path) -> Result<()> {
        // same as in vdj/model.rs, not ideal
        let path = directory.join("model_params.txt");
        let mut file = File::create(path)?;
        let params = self.write_params()?;
        file.write_all(params.as_bytes())?;

        let path = directory.join("model_marginals.txt");
        let mut file = File::create(path)?;
        let marginals = self.write_marginals()?;
        file.write_all(marginals.as_bytes())?;

        let path = directory.join("V_gene_CDR3_anchors.csv");
        let mut file = File::create(path)?;
        let vanchors = self.write_v_anchors()?;
        file.write_all(vanchors.as_bytes())?;

        let path = directory.join("J_gene_CDR3_anchors.csv");
        let mut file = File::create(path)?;
        let janchors = self.write_j_anchors()?;
        file.write_all(janchors.as_bytes())?;

        Ok(())
    }

    pub fn write_v_anchors(&self) -> Result<String> {
        self.inner.write_v_anchors()
    }

    pub fn write_j_anchors(&self) -> Result<String> {
        self.inner.write_j_anchors()
    }

    pub fn write_marginals(&self) -> Result<String> {
        let marginal_vs = Marginal::create(Vec::new(), self.p_v.clone().into_dyn()).write()?;
        let marginal_js = Marginal::create(
            vec!["v_choice"],
            self.p_j_given_v.clone().permuted_axes((1, 0)).into_dyn(),
        )
        .write()?;
        let marginal_delv = Marginal::create(
            vec!["v_choice"],
            self.p_del_v_given_v
                .clone()
                .permuted_axes((1, 0))
                .into_dyn(),
        )
        .write()?;
        let marginal_delj = Marginal::create(
            vec!["j_choice"],
            self.p_del_j_given_j
                .clone()
                .permuted_axes((1, 0))
                .into_dyn(),
        )
        .write()?;

        let marginal_vjins =
            Marginal::create(Vec::new(), self.p_ins_vj.clone().into_dyn()).write()?;
        let marginal_vjdinucl = Marginal::create(
            Vec::new(),
            self.markov_coefficients_vj
                .iter()
                .cloned()
                .collect::<Array1<f64>>()
                .into_dyn(),
        )
        .write()?;

        Ok(format!(
            "@v_choice\n\
	     {marginal_vs}\
	     @j_choice\n\
	     {marginal_js}\
	     @v_3_del\n\
	     {marginal_delv}\
	     @j_5_del\n\
	     {marginal_delj}\
	     @vj_ins\n\
	     {marginal_vjins}\
	     @vj_dinucl\n\
	     {marginal_vjdinucl}\
	     "
        ))
    }

    pub fn write_params(&self) -> Result<String> {
        let mut result = "@Event_list\n\
			  #GeneChoice;V_gene;Undefined_side;7;v_choice\n"
            .to_string();
        let vgenes = EventType::Genes(self.seg_vs.clone());
        result.push_str(&vgenes.write());

        result.push_str("#GeneChoice;J_gene;Undefined_side;6;j_choice\n");
        let jgenes = EventType::Genes(self.seg_js.clone());
        result.push_str(&jgenes.write());

        result.push_str("#Deletion;V_gene;Three_prime;5;v_3_del\n");
        let delvs = EventType::Numbers((self.range_del_v.0..self.range_del_v.1 + 1).collect());
        result.push_str(&delvs.write());

        result.push_str("#Deletion;J_gene;Five_prime;5;j_5_del\n");
        let deljs = EventType::Numbers((self.range_del_j.0..self.range_del_j.1 + 1).collect());
        result.push_str(&deljs.write());

        result.push_str("#Insertion;VJ_gene;Undefined_side;4;vj_ins\n");
        let insvjs = EventType::Numbers((0 as i64..self.p_ins_vj.dim() as i64).collect());
        result.push_str(&insvjs.write());

        let dimv = self.seg_vs.len();
        let dimdelv = self.p_del_v_given_v.dim().0;
        let dimj = self.seg_js.len();
        let dimdelj = self.p_del_j_given_j.dim().0;
        let error_rate = self.error_rate;
        result.push_str(&format!(
            "#DinucMarkov;VJ_gene;Undefined_side;3;vj_dinucl\n\
	     %T;3\n\
	     %C;1\n\
	     %G;2\n\
	     %A;0\n\
	     @Edges\n\
	     %GeneChoice_V_gene_Undefined_side_prio7_size{dimv};\
	     GeneChoice_J_gene_Undefined_side_prio6_size{dimj}\n\
	     %GeneChoice_V_gene_Undefined_side_prio7_size{dimv};\
	     Deletion_V_gene_Three_prime_prio5_size{dimdelv}\n\
	     %GeneChoice_J_gene_Undefined_side_prio6_size{dimj};\
	     Deletion_J_gene_Five_prime_prio5_size{dimdelj}\n\
	     @ErrorRate\n\
	     #SingleErrorRate\n\
	     {error_rate}\n"
        ));
        Ok(result)
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
        self.update_outer_model()?;

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
        Ok(())
    }

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

    pub fn generate_without_errors<R: Rng>(
        &mut self,
        functional: bool,
        rng: &mut R,
    ) -> GenerationResult {
        let gen_result = self.inner.generate_without_errors(functional, rng);
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
        (self.get_p_vj()).sum_axis(Axis(0))
    }

    pub fn get_p_vj(&self) -> Array2<f64> {
        (self.p_j_given_v.clone() * self.p_v.clone()).t().to_owned()
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

    pub fn align_from_cdr3(
        &self,
        cdr3_seq: Dna,
        vgenes: Vec<Gene>,
        jgenes: Vec<Gene>,
    ) -> Result<Sequence> {
        self.inner.align_from_cdr3(cdr3_seq, vgenes, jgenes)
    }

    pub fn align_sequence(
        &self,
        dna_seq: Dna,
        align_params: &AlignmentParameters,
    ) -> Result<Sequence> {
        self.inner.align_sequence(dna_seq, align_params)
    }

    pub fn recreate_full_sequence(&self, dna: &Dna, vgene: &Gene, jgene: &Gene) -> Dna {
        self.inner.recreate_full_sequence(dna, vgene, jgene)
    }

    pub fn from_features(&self, feature: &Features) -> Result<Model> {
        let mut m = self.clone();
        m.inner.update(feature)?;
        m.update_outer_model()?;
        m.initialize()?;
        Ok(m)
    }

    pub fn similar_to(&self, m: Model) -> bool {
        self.inner.similar_to(m.inner)
    }
}

impl ModelGen for Model {
    fn get_v_segments(&self) -> Vec<Gene> {
        self.seg_vs.clone()
    }
    fn get_j_segments(&self) -> Vec<Gene> {
        self.seg_js.clone()
    }
}
