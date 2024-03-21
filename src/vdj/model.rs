use crate::sequence::{
    utils::count_differences, utils::NUCLEOTIDES, AlignmentParameters, AminoAcid, Dna,
};
use crate::sequence::{DAlignment, VJAlignment};
use crate::shared::model::{sanitize_j, sanitize_v};
use crate::shared::parser::{
    parse_file, parse_str, EventType, Marginal, ParserMarginals, ParserParams,
};
use crate::shared::utils::{
    calc_steady_state_dist, sorted_and_complete, sorted_and_complete_0start, DiscreteDistribution,
    ErrorDistribution, Gene, InferenceParameters, MarkovDNA, ModelGen, Normalize, RecordModel,
};
use crate::vdj::inference::ResultInference;
use crate::vdj::sequence::{align_all_dgenes, align_all_jgenes, align_all_vgenes};
use crate::vdj::{Features, InfEvent, Sequence, StaticEvent};
use anyhow::{anyhow, Result};
use ndarray::{s, Array1, Array2, Array3, Axis};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::Distribution;
use rayon::prelude::*;
use std::path::Path;
use std::{cmp, fs::read_to_string, fs::File, io::Write};

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct Generator {
    model: Model,
    rng: SmallRng,
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

impl Generator {
    /// available_v, available_j: set of v,j genes to choose from
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

#[derive(Default, Clone, Debug)]
pub struct Generative {
    // Contains the distribution needed to generate the model
    d_vdj: DiscreteDistribution,
    d_ins_vd: DiscreteDistribution,
    d_ins_dj: DiscreteDistribution,
    d_del_v_given_v: Vec<DiscreteDistribution>,
    d_del_j_given_j: Vec<DiscreteDistribution>,
    d_del_d5_del_d3: Vec<DiscreteDistribution>,
    markov_vd: MarkovDNA,
    markov_dj: MarkovDNA,
    error: ErrorDistribution,
}

#[derive(Default, Clone, Debug)]
pub struct Model {
    // Sequence information
    pub seg_vs: Vec<Gene>,
    pub seg_js: Vec<Gene>,
    pub seg_ds: Vec<Gene>,

    // V/J sequences trimmed at the CDR3 region (include F/W/C residues) with
    // the maximum number of reverse palindromic insertions appended.
    pub seg_vs_sanitized: Vec<Dna>, // match genomic_data.cutV_genomic_CDR3_seqs
    pub seg_js_sanitized: Vec<Dna>,

    // Probabilities of the different events
    pub p_vdj: Array3<f64>,
    pub p_ins_vd: Array1<f64>,
    pub p_ins_dj: Array1<f64>,
    pub p_del_v_given_v: Array2<f64>,
    pub p_del_j_given_j: Array2<f64>,
    pub p_del_d5_del_d3: Array3<f64>, // P(del_d5, del_d3 | D)
    pub gen: Generative,
    pub markov_coefficients_vd: Array2<f64>,
    pub markov_coefficients_dj: Array2<f64>,
    pub range_del_v: (i64, i64),
    pub range_del_j: (i64, i64),
    pub range_del_d3: (i64, i64),
    pub range_del_d5: (i64, i64),
    pub error_rate: f64,

    // Not directly useful for the model but useful for integration with other soft
    // TODO: transform these in "getter" in the python bindings, they don't need to be precomputed
    pub p_v: Array1<f64>,
    pub p_dj: Array2<f64>,
    pub p_d_given_vj: Array3<f64>,
    pub p_j_given_v: Array2<f64>,
    pub first_nt_bias_ins_vd: Array1<f64>,
    pub first_nt_bias_ins_dj: Array1<f64>,
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
        let mut wtr = csv::Writer::from_writer(vec![]);
        wtr.write_record(&["gene", "anchor_index", "function"])?;
        for gene in &self.seg_vs {
            let cdr3_pos = format!(
                "{}",
                gene.cdr3_pos.ok_or(anyhow!("Corrupted Model struct."))?
            );
            wtr.write_record(&[gene.name.clone(), cdr3_pos, gene.functional.clone()])?;
        }
        wtr.flush()?;
        let data = String::from_utf8(wtr.into_inner()?)?;
        Ok(data)
    }

    pub fn write_j_anchors(&self) -> Result<String> {
        let mut wtr = csv::Writer::from_writer(vec![]);
        wtr.write_record(&["gene", "anchor_index", "function"])?;
        for gene in &self.seg_js {
            let cdr3_pos = format!(
                "{}",
                gene.cdr3_pos.ok_or(anyhow!("Corrupted Model struct."))?
            );
            wtr.write_record(&[gene.name.clone(), cdr3_pos, gene.functional.clone()])?;
        }
        wtr.flush()?;
        let data = String::from_utf8(wtr.into_inner()?)?;
        Ok(data)
    }

    pub fn write_marginals(&self) -> Result<String> {
        let marginal_vs = Marginal::create(Vec::new(), self.p_v.clone().into_dyn()).write()?;
        let marginal_js = Marginal::create(
            vec!["v_choice"],
            self.p_j_given_v.clone().permuted_axes((1, 0)).into_dyn(),
        )
        .write()?;
        let marginal_ds = Marginal::create(
            vec!["v_choice", "j_choice"],
            self.p_d_given_vj
                .clone()
                .permuted_axes((1, 2, 0))
                .into_dyn(),
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

        let p_d3_d5_d = self.p_del_d5_del_d3.clone().permuted_axes((1, 0, 2));
        let p_deld5_given_d = p_d3_d5_d.sum_axis(Axis(0));
        let p_deld3_given_deld5_d = (p_d3_d5_d.clone()
            / &p_deld5_given_d.broadcast(p_d3_d5_d.clone().dim()).unwrap())
            .mapv(|x| if x.is_nan() { 0.0 } else { x });

        let marginal_deld5 = Marginal::create(
            vec!["d_gene"],
            p_deld5_given_d.permuted_axes((1, 0)).into_dyn(),
        )
        .write()?;
        let marginal_deld3 = Marginal::create(
            vec!["d_gene", "d_5_del"],
            p_deld3_given_deld5_d.permuted_axes((2, 1, 0)).into_dyn(),
        )
        .write()?;
        let marginal_vdins =
            Marginal::create(Vec::new(), self.p_ins_vd.clone().into_dyn()).write()?;
        let marginal_vddinucl = Marginal::create(
            Vec::new(),
            self.markov_coefficients_vd
                .iter()
                .cloned()
                .collect::<Array1<f64>>()
                .into_dyn(),
        )
        .write()?;
        let marginal_djins =
            Marginal::create(Vec::new(), self.p_ins_dj.clone().into_dyn()).write()?;
        let marginal_djdinucl = Marginal::create(
            Vec::new(),
            self.markov_coefficients_dj
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
	     @d_gene\n\
	     {marginal_ds}\
	     @v_3_del\n\
	     {marginal_delv}\
	     @d_5_del\n\
	     {marginal_deld5}\
	     @d_3_del\n\
	     {marginal_deld3}\
	     @j_5_del\n\
	     {marginal_delj}\
	     @vd_ins\n\
	     {marginal_vdins}\
	     @vd_dinucl\n\
	     {marginal_vddinucl}\
	     @dj_ins\n\
	     {marginal_djins}\
	     @dj_dinucl\n\
	     {marginal_djdinucl}"
        ))
    }

    pub fn write_params(&self) -> Result<String> {
        let mut result = "@Event_list\n\
			  #GeneChoice;V_gene;Undefined_side;7;v_choice\n"
            .to_string();
        let vgenes = EventType::Genes(self.seg_vs.clone());
        result.push_str(&vgenes.write());

        result.push_str("#GeneChoice;D_gene;Undefined_side;6;d_gene\n");
        let dgenes = EventType::Genes(self.seg_ds.clone());
        result.push_str(&dgenes.write());

        result.push_str("#GeneChoice;J_gene;Undefined_side;7;j_choice\n");
        let jgenes = EventType::Genes(self.seg_js.clone());
        result.push_str(&jgenes.write());

        result.push_str("#Deletion;V_gene;Three_prime;5;v_3_del\n");
        let delvs = EventType::Numbers((self.range_del_v.0..self.range_del_v.1 + 1).collect());
        result.push_str(&delvs.write());

        result.push_str("#Deletion;D_gene;Three_prime;5;d_3_del\n");
        let deld3s = EventType::Numbers((self.range_del_d3.0..self.range_del_d3.1 + 1).collect());
        result.push_str(&deld3s.write());

        result.push_str("#Deletion;D_gene;Five_prime;5;d_5_del\n");
        let deld5s = EventType::Numbers((self.range_del_d5.0..self.range_del_d5.1 + 1).collect());
        result.push_str(&deld5s.write());

        result.push_str("#Deletion;J_gene;Five_prime;5;j_5_del\n");
        let deljs = EventType::Numbers((self.range_del_j.0..self.range_del_j.1 + 1).collect());
        result.push_str(&deljs.write());

        result.push_str("#Insertion;VD_genes;Undefined_side;4;vd_ins\n");
        let insvds = EventType::Numbers((0 as i64..self.p_ins_vd.dim() as i64).collect());
        result.push_str(&insvds.write());

        result.push_str("#Insertion;DJ_genes;Undefined_side;2;dj_ins\n");
        let insdjs = EventType::Numbers((0 as i64..self.p_ins_dj.dim() as i64).collect());
        result.push_str(&insdjs.write());

        let dimv = self.seg_vs.len();
        let dimdelv = self.p_del_v_given_v.dim().0;
        let dimd = self.seg_ds.len();
        let dimdeld3 = self.p_del_d5_del_d3.dim().1;
        let dimdeld5 = self.p_del_d5_del_d3.dim().0;
        let dimj = self.seg_js.len();
        let dimdelj = self.p_del_j_given_j.dim().0;
        let error_rate = self.error_rate;
        result.push_str(&format!(
            "#DinucMarkov;VD_genes;Undefined_side;3;vd_dinucl\n\
	     %T;3\n\
	     %C;1\n\
	     %G;2\n\
	     %A;0\n\
	     #DinucMarkov;DJ_gene;Undefined_side;1;dj_dinucl\n\
	     %T;3\n\
	     %C;1\n\
	     %G;2\n\
	     %A;0\n\
	     @Edges\n\
	     %GeneChoice_V_gene_Undefined_side_prio7_size{dimv};\
	     Deletion_V_gene_Three_prime_prio5_size{dimdelv}\n\
	     %GeneChoice_D_gene_Undefined_side_prio6_size{dimd};\
	     Deletion_D_gene_Three_prime_prio5_size{dimdeld3}\n\
	     %GeneChoice_D_gene_Undefined_side_prio6_size{dimd};\
	     Deletion_D_gene_Five_prime_prio5_size{dimdeld5}\n\
	     %GeneChoice_J_gene_Undefined_side_prio7_size{dimj};\
	     Deletion_J_gene_Five_prime_prio5_size{dimdelj}\n\
	     %GeneChoice_J_gene_Undefined_side_prio7_size{dimj};\
	     GeneChoice_D_gene_Undefined_side_prio6_size{dimd}\n\
	     %Deletion_D_gene_Five_prime_prio5_size{dimdeld5};\
	     Deletion_D_gene_Three_prime_prio5_size{dimdeld3}\n\
	     @ErrorRate\n\
	     #SingleErrorRate\n\
	     {error_rate}\n"
        ));
        Ok(result)
    }

    pub fn load_model(pp: &ParserParams, pm: &ParserMarginals) -> Result<Model> {
        let mut model = Model {
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
            seg_ds: pp
                .params
                .get("d_gene")
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
            (*arrdelv.iter().min().ok_or(anyhow!("Empty v_3_del"))?),
            (*arrdelv.iter().max().ok_or(anyhow!("Empty v_3_del"))?),
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
        let arrdeld3 = pp
            .params
            .get("d_3_del")
            .ok_or(anyhow!("Invalid d_3_del"))?
            .clone()
            .to_numbers()?;

        model.range_del_d3 = (
            *arrdeld3.iter().min().ok_or(anyhow!("Empty d_3_del"))?,
            *arrdeld3.iter().max().ok_or(anyhow!("Empty d_3_del"))?,
        );

        let arrdeld5 = pp
            .params
            .get("d_5_del")
            .ok_or(anyhow!("Invalid d_5_del"))?
            .clone()
            .to_numbers()?;

        model.range_del_d5 = (
            *arrdeld5.iter().min().ok_or(anyhow!("Empty d_5_del"))?,
            *arrdeld5.iter().max().ok_or(anyhow!("Empty d_5_del"))?,
        );

        model.sanitize_genes()?;

        if !(sorted_and_complete(arrdelv)
            & sorted_and_complete(arrdelj)
            & sorted_and_complete(arrdeld3)
            & sorted_and_complete(arrdeld5)
            & sorted_and_complete_0start(
                pp.params
                    .get("vd_ins")
                    .ok_or(anyhow!("Invalid vd_ins"))?
                    .clone()
                    .to_numbers()?,
            )
            & sorted_and_complete_0start(
                pp.params
                    .get("dj_ins")
                    .ok_or(anyhow!("Invalid dj_ins"))?
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
        let pv = pm.marginals.get("v_choice").unwrap().probabilities.clone();
        let pd = pm.marginals.get("d_gene").unwrap().probabilities.clone();
        let pj = pm.marginals.get("j_choice").unwrap().probabilities.clone();

        // For P(V, D, J) two possibilities;
        // P(V) P(D,J) [olga model] and P(V,D,J) [igor model]
        // Igor-like
        if pm.marginals.get("v_choice").unwrap().dimensions.len() == 1
            && pm.marginals.get("j_choice").unwrap().dimensions.len() == 2
            && pm.marginals.get("d_gene").unwrap().dimensions.len() == 3
        {
            let vdim = pd.dim()[0];
            let jdim = pd.dim()[1];
            let ddim = pd.dim()[2];
            model.p_vdj = Array3::<f64>::zeros((vdim, ddim, jdim));
            for vv in 0..vdim {
                for jj in 0..jdim {
                    for dd in 0..ddim {
                        model.p_vdj[[vv, dd, jj]] = pd[[vv, jj, dd]] * pj[[vv, jj]] * pv[[vv]];
                    }
                }
            }
        }
        // Olga-like
        else if pm.marginals.get("v_choice").unwrap().dimensions.len() == 1
            && pm.marginals.get("j_choice").unwrap().dimensions.len() == 1
            && pm.marginals.get("d_gene").unwrap().dimensions.len() == 2
        {
            let vdim = pv.dim()[0];
            let jdim = pd.dim()[0];
            let ddim = pd.dim()[1];
            model.p_vdj = Array3::<f64>::zeros((vdim, ddim, jdim));
            for vv in 0..vdim {
                for jj in 0..jdim {
                    for dd in 0..ddim {
                        model.p_vdj[[vv, dd, jj]] = pd[[jj, dd]] * pv[[vv]] * pj[[jj]];
                    }
                }
            }
        } else {
            return Err::<Model, anyhow::Error>(anyhow!("Wrong format for the VDJ probabilities"));
        }

        model.set_p_vdj(&model.p_vdj.clone())?;

        model.p_ins_vd = pm
            .marginals
            .get("vd_ins")
            .unwrap()
            .probabilities
            .clone()
            .into_dimensionality()
            .unwrap();
        model.p_ins_dj = pm
            .marginals
            .get("dj_ins")
            .unwrap()
            .probabilities
            .clone()
            .into_dimensionality()
            .unwrap();
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
        if (model.range_del_v.1 - model.range_del_v.0 + 1) != (model.p_del_v_given_v.dim().0 as i64)
        {
            return Err(anyhow!("Wrong format for V deletions"));
        }
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

        if (model.range_del_j.1 - model.range_del_j.0 + 1) != (model.p_del_j_given_j.dim().0 as i64)
        {
            return Err(anyhow!("Wrong format for J deletions"));
        }

        // compute the joint probability P(delD3, delD5 | D)
        // P(delD3, delD5 | D) = P(delD3 | delD5, D) * P(delD5 | D)
        let pdeld3 = pm.marginals.get("d_3_del").unwrap().probabilities.clone(); // P(deld3| delD5, D)
        let pdeld5 = pm.marginals.get("d_5_del").unwrap().probabilities.clone(); // P(deld5| D)
        let ddim = pdeld3.dim()[0];
        let d5dim = pdeld3.dim()[1];
        let d3dim = pdeld3.dim()[2];
        model.p_del_d5_del_d3 = Array3::<f64>::zeros((d5dim, d3dim, ddim));
        for dd in 0..ddim {
            for d5 in 0..d5dim {
                for d3 in 0..d3dim {
                    model.p_del_d5_del_d3[[d5, d3, dd]] = pdeld3[[dd, d5, d3]] * pdeld5[[dd, d5]]
                }
            }
        }

        if (model.range_del_d5.1 - model.range_del_d5.0 + 1)
            != (model.p_del_d5_del_d3.dim().0 as i64)
        {
            return Err(anyhow!("Wrong format for D5 deletions"));
        }

        if (model.range_del_d3.1 - model.range_del_d3.0 + 1)
            != (model.p_del_d5_del_d3.dim().1 as i64)
        {
            return Err(anyhow!("Wrong format for D3 deletions"));
        }

        // Markov coefficients
        model.markov_coefficients_vd = pm
            .marginals
            .get("vd_dinucl")
            .unwrap()
            .probabilities
            .clone()
            .into_shape((4, 4))
            .map_err(|_e| anyhow!("Wrong size for vd_dinucl"))?;
        model.markov_coefficients_dj = pm
            .marginals
            .get("dj_dinucl")
            .unwrap()
            .probabilities
            .clone()
            .into_shape((4, 4))
            .map_err(|_e| anyhow!("Wrong size for dj_dinucl"))?;

        // TODO: Need to deal with potential first nt bias in the file
        model.first_nt_bias_ins_vd =
            Array1::from_vec(calc_steady_state_dist(&model.markov_coefficients_vd)?);
        model.first_nt_bias_ins_dj =
            Array1::from_vec(calc_steady_state_dist(&model.markov_coefficients_dj)?);

        model.error_rate = pp.error_rate;
        model.thymic_q = 9.41; // TODO: deal with this

        model.initialize()?;

        Ok(model)
    }

    pub fn sanitize_genes(&mut self) -> Result<()> {
        // Trim the V/J nucleotides sequences at the CDR3 region (include F/W/C residues)
        // and append the maximum number of reverse palindromic insertions appended.

        // Add the palindromic insertions
        for g in self.seg_vs.iter_mut() {
            g.create_palindromic_ends(0, (-self.range_del_v.0) as usize);
        }
        for g in self.seg_js.iter_mut() {
            g.create_palindromic_ends((-self.range_del_j.0) as usize, 0);
        }

        for g in self.seg_ds.iter_mut() {
            g.create_palindromic_ends(
                (-self.range_del_d5.0) as usize,
                (-self.range_del_d3.0) as usize,
            );
        }

        // cut the V/J at the CDR3 region
        self.seg_vs_sanitized = sanitize_v(self.seg_vs.clone())?;
        self.seg_js_sanitized = sanitize_j(self.seg_js.clone(), (-self.range_del_j.0) as usize)?;
        Ok(())
    }

    fn initialize_generative_model(&mut self) -> Result<()> {
        self.gen.d_vdj = DiscreteDistribution::new(self.p_vdj.view().iter().cloned().collect())?;
        self.gen.d_ins_vd = DiscreteDistribution::new(self.p_ins_vd.to_vec())?;
        self.gen.d_ins_dj = DiscreteDistribution::new(self.p_ins_dj.to_vec())?;

        self.gen.d_del_v_given_v = Vec::new();
        for row in self.p_del_v_given_v.axis_iter(Axis(1)) {
            self.gen
                .d_del_v_given_v
                .push(DiscreteDistribution::new(row.to_vec())?);
        }
        self.gen.d_del_j_given_j = Vec::new();
        for row in self.p_del_j_given_j.axis_iter(Axis(1)) {
            self.gen
                .d_del_j_given_j
                .push(DiscreteDistribution::new(row.to_vec())?);
        }

        self.gen.d_del_d5_del_d3 = Vec::new();
        for ddd in 0..self.p_del_d5_del_d3.dim().2 {
            let d5d3: Vec<f64> = self
                .p_del_d5_del_d3
                .slice(s![.., .., ddd])
                .iter()
                .cloned()
                .collect();
            self.gen
                .d_del_d5_del_d3
                .push(DiscreteDistribution::new(d5d3)?);
        }

        self.gen.markov_vd = MarkovDNA::new(self.markov_coefficients_vd.to_owned())?;
        self.gen.markov_dj = MarkovDNA::new(self.markov_coefficients_dj.to_owned())?;

        self.gen.error = Default::default();

        Ok(())
    }

    /// Return (full_seq, cdr3_seq, aa_seq, event)
    pub fn generate_no_error<R: Rng>(
        &mut self,
        functional: bool,
        rng: &mut R,
    ) -> (Dna, Dna, Option<AminoAcid>, StaticEvent) {
        // loop until we find a valid sequence (if generating functional alone)
        loop {
            let mut event = StaticEvent {
                ..Default::default()
            };

            let vdj_index: usize = self.gen.d_vdj.generate(rng);
            event.v_index = vdj_index / (self.p_vdj.dim().1 * self.p_vdj.dim().2);
            event.d_index =
                (vdj_index % (self.p_vdj.dim().1 * self.p_vdj.dim().2)) / self.p_vdj.dim().2;
            event.j_index = vdj_index % self.p_dj.dim().1;

            let seq_v_cdr3: &Dna = &self.seg_vs_sanitized[event.v_index];
            let seq_j_cdr3: &Dna = &self.seg_js_sanitized[event.j_index];

            let seq_d: &Dna = self.seg_ds[event.d_index].seq_with_pal.as_ref().unwrap();
            let seq_v: &Dna = self.seg_vs[event.v_index].seq_with_pal.as_ref().unwrap();
            let seq_j: &Dna = self.seg_js[event.j_index].seq_with_pal.as_ref().unwrap();

            event.delv = self.gen.d_del_v_given_v[event.v_index].generate(rng);
            let del_d: usize = self.gen.d_del_d5_del_d3[event.d_index].generate(rng);
            event.deld5 = del_d / self.p_del_d5_del_d3.dim().1;
            event.deld3 = del_d % self.p_del_d5_del_d3.dim().1;
            event.delj = self.gen.d_del_j_given_j[event.j_index].generate(rng);

            let ins_vd: usize = self.gen.d_ins_vd.generate(rng);
            let ins_dj: usize = self.gen.d_ins_dj.generate(rng);

            // println!(
            //     "{:?}",
            //     (
            //         seq_v_cdr3.get_string(),
            //         seq_j_cdr3.get_string(),
            //         event.delv,
            //         seq_d.get_string(),
            //         event.deld5,
            //         event.deld3,
            //         event.delj,
            //         ins_vd,
            //         ins_dj,
            //     ),
            // );

            let out_of_frame = (seq_v_cdr3.len() + seq_j_cdr3.len() - event.delv + seq_d.len()
                - event.deld5
                - event.deld3
                - event.delj
                + ins_vd
                + ins_dj)
                % 3
                != 0;
            if functional && out_of_frame {
                continue;
            }

            // look at the last nucleotide of V (for the Markov chain)

            let end_v = seq_v.seq[seq_v.len() - event.delv - 1];
            let first_j = seq_j.seq[event.delj];

            let ins_seq_vd: Dna = self.gen.markov_vd.generate(ins_vd, end_v, rng);
            let mut ins_seq_dj: Dna = self.gen.markov_dj.generate(ins_dj, first_j, rng);
            ins_seq_dj.reverse(); // reverse for integration

            event.insdj = ins_seq_dj.clone();
            event.insvd = ins_seq_vd.clone();
            event.v_start_gene = 0;
            event.d_start_seq = seq_v.len() - event.delv - event.deld5 + ins_vd;
            event.j_start_seq = seq_v.len() - event.delv + ins_vd + ins_dj + seq_d.len()
                - event.deld5
                - event.deld3
                - event.delj;

            // create the complete sequence:
            let full_seq = event.to_sequence(self);

            // println!("{:?}", full_seq.get_string());

            // create the complete CDR3 sequence
            let cdr3_seq = event.to_cdr3(self);

            // translate
            let cdr3_seq_aa: Option<AminoAcid> = cdr3_seq.translate().ok();

            match cdr3_seq_aa {
                Some(saa) => {
                    // check for stop codon
                    if functional && saa.seq.contains(&b'*') {
                        continue;
                    }

                    // check for conserved extremities (cysteine)
                    if functional && (saa.seq[0] != b'C') {
                        continue;
                    }
                    return (full_seq, cdr3_seq, Some(saa), event);
                }
                None => {
                    if functional {
                        continue;
                    }
                    return (full_seq, cdr3_seq, None, event);
                }
            }
        }
    }

    /// Return (cdr3_nt, cdr3_aa, full_sequence, event, vname, jname)
    pub fn generate<R: Rng>(&mut self, functional: bool, rng: &mut R) -> GenerationResult {
        let (full_seq, _, _, mut event) = self.generate_no_error(functional, rng);

        // add errors
        let effective_error_rate = self.error_rate * 4. / 3.;
        event.errors =
            Vec::with_capacity((effective_error_rate * full_seq.len() as f64).ceil() as usize);

        for (idx, nucleotide) in full_seq.seq.iter().enumerate() {
            if self.gen.error.is_error.sample(rng) < effective_error_rate {
                let a = NUCLEOTIDES[self.gen.error.nucleotide.sample(rng)];
                if a != *nucleotide {
                    event.errors.push((idx, a));
                }
            }
        }

        let full_seq = event.to_sequence(&self);
        let cdr3_nt = event.extract_cdr3(&full_seq, &self);
        let cdr3_aa = cdr3_nt.translate().ok();

        GenerationResult {
            v_gene: self.seg_vs[event.v_index].name.clone(),
            j_gene: self.seg_js[event.j_index].name.clone(),
            recombination_event: event,
            full_seq: full_seq.get_string(),
            cdr3_nt: cdr3_nt.get_string(),
            cdr3_aa: cdr3_aa.map(|x| x.to_string()),
        }
    }

    /// Return (cdr3_nt, cdr3_aa, full_sequence, event, vname, jname)
    pub fn generate_without_errors<R: Rng>(
        &mut self,
        functional: bool,
        rng: &mut R,
    ) -> GenerationResult {
        let (full_seq, cdr3_nt, cdr3_aa, event) = self.generate_no_error(functional, rng);

        GenerationResult {
            v_gene: self.seg_vs[event.v_index].name.clone(),
            j_gene: self.seg_js[event.j_index].name.clone(),
            recombination_event: event,
            full_seq: full_seq.get_string(),
            cdr3_nt: cdr3_nt.get_string(),
            cdr3_aa: cdr3_aa.map(|x| x.to_string()),
        }
    }

    pub fn uniform(&self) -> Result<Model> {
        let mut m = Model {
            seg_vs: self.seg_vs.clone(),
            seg_js: self.seg_js.clone(),
            seg_ds: self.seg_ds.clone(),

            range_del_d3: self.range_del_d3,
            range_del_v: self.range_del_v,
            range_del_j: self.range_del_j,
            range_del_d5: self.range_del_d5,
            p_vdj: Array3::<f64>::ones(self.p_vdj.dim()),
            p_j_given_v: Array2::<f64>::ones(self.p_j_given_v.dim()),
            p_d_given_vj: Array3::<f64>::ones(self.p_d_given_vj.dim()),
            p_ins_vd: Array1::<f64>::ones(self.p_ins_vd.dim()),
            p_ins_dj: Array1::<f64>::ones(self.p_ins_dj.dim()),
            p_del_v_given_v: Array2::<f64>::ones(self.p_del_v_given_v.dim()),
            p_del_j_given_j: Array2::<f64>::ones(self.p_del_j_given_j.dim()),
            p_del_d5_del_d3: Array3::<f64>::ones(self.p_del_d5_del_d3.dim()),
            markov_coefficients_vd: Array2::<f64>::ones(self.markov_coefficients_vd.dim()),
            markov_coefficients_dj: Array2::<f64>::ones(self.markov_coefficients_dj.dim()),
            first_nt_bias_ins_vd: Array1::<f64>::ones(self.first_nt_bias_ins_vd.dim()),
            first_nt_bias_ins_dj: Array1::<f64>::ones(self.first_nt_bias_ins_dj.dim()),
            error_rate: 0.1, // TODO: too ad-hoc
            ..Default::default()
        };
        m.initialize()?;
        Ok(m)
    }

    pub fn initialize(&mut self) -> Result<()> {
        self.sanitize_genes()?;
        // load the data and normalize
        let mut feature = Features::new(self)?;
        feature.normalize()?;
        self.load_features(&feature)?;
        self.initialize_generative_model()?;
        Ok(())
    }

    pub fn evaluate(
        &self,
        sequence: &Sequence,
        inference_params: &InferenceParameters,
    ) -> Result<ResultInference> {
        let mut feature = Features::new(self)?;
        let mut result = feature.infer(sequence, inference_params)?;
        result.fill_event(self, sequence)?;

        // compute the pgen if needed
        if self.error_rate == 0. {
            // no error: likelihood = pgen
            result.pgen = result.likelihood;
        }

        if inference_params.compute_pgen && inference_params.store_best_event {
            // if there is error, we use the reconstructed sequence to infer everything
            let event = result
                .get_best_event()
                .ok_or(anyhow!("Error with pgen inference"))?;
            let seq_without_err = event
                .reconstructed_sequence
                .ok_or(anyhow!("Error with pgen inference"))?;
            // full sequence, so default alignment parameters should be fine.
            let aligned_seq =
                self.align_sequence(seq_without_err, &AlignmentParameters::default())?;
            let mut feature = Features::new(self)?;
            feature.error.error_rate = 0.; // remove the error
            result.pgen = feature.infer(&aligned_seq, inference_params)?.likelihood;
        }

        result.features = Some(feature.clone());
        Ok(result)
    }

    pub fn infer(
        &mut self,
        sequences: &Vec<Sequence>,
        inference_params: &InferenceParameters,
    ) -> Result<()> {
        let mut ip = inference_params.clone();
        ip.infer = true;
        ip.compute_pgen = false;
        ip.store_best_event = false;
        let features = sequences
            .par_iter()
            .map(|sequence| {
                let mut feature = Features::new(self)?;
                let _ = feature.infer(sequence, &ip)?;
                Ok(feature)
            })
            .collect::<Result<Vec<_>>>()?;

        let avg_features = Features::average(features)?;
        self.update(&avg_features)?;
        Ok(())
    }

    pub fn filter_vs(&self, vs: Vec<Gene>) -> Result<Model> {
        let mut m = self.clone();

        let dim = self.p_vdj.dim();
        m.p_vdj = Array3::<f64>::zeros((vs.len(), dim.1, dim.2));
        m.seg_vs = Vec::new();
        m.p_del_v_given_v = Array2::<f64>::zeros((self.p_del_v_given_v.dim().0, vs.len()));

        let mut iv_restr = 0;
        for iv in 0..dim.0 {
            let vgene = self.seg_vs[iv].clone();
            if vs.contains(&vgene) {
                m.seg_vs.push(vgene);
                for id in 0..dim.1 {
                    for ij in 0..dim.2 {
                        m.p_vdj[[iv_restr, id, ij]] = self.p_vdj[[iv, id, ij]];
                    }
                }
                for idelv in 0..self.p_del_v_given_v.dim().0 {
                    m.p_del_v_given_v[[idelv, iv_restr]] = self.p_del_v_given_v[[idelv, iv]];
                }
                iv_restr += 1;
            }
        }
        m.initialize()?;
        Ok(m)
    }

    pub fn filter_js(&self, js: Vec<Gene>) -> Result<Model> {
        let mut m = self.clone();
        let dim = self.p_vdj.dim();

        m.p_vdj = Array3::<f64>::zeros((dim.0, dim.1, js.len()));
        m.seg_js = Vec::new();
        m.p_del_j_given_j = Array2::<f64>::zeros((self.p_del_j_given_j.dim().0, js.len()));

        let mut ij_restr = 0;
        for ij in 0..dim.2 {
            let jgene = self.seg_js[ij].clone();
            if js.contains(&jgene) {
                m.seg_js.push(jgene);
                for id in 0..dim.1 {
                    for iv in 0..dim.0 {
                        m.p_vdj[[iv, id, ij_restr]] = self.p_vdj[[iv, id, ij]];
                    }
                }
                for idelj in 0..self.p_del_j_given_j.dim().0 {
                    m.p_del_j_given_j[[idelj, ij_restr]] = self.p_del_j_given_j[[idelj, ij]];
                }
                ij_restr += 1;
            }
        }
        m.initialize()?;
        Ok(m)
    }

    pub fn get_v_gene(&self, event: &InfEvent) -> String {
        self.seg_vs[event.v_index].name.clone()
    }

    pub fn get_j_gene(&self, event: &InfEvent) -> String {
        self.seg_js[event.j_index].name.clone()
    }

    pub fn get_d_gene(&self, event: &InfEvent) -> String {
        self.seg_ds[event.d_index].name.clone()
    }

    pub fn align_from_cdr3(
        &self,
        cdr3_seq: Dna,
        vgenes: Vec<Gene>,
        jgenes: Vec<Gene>,
    ) -> Result<Sequence> {
        let v_alignments = vgenes
            .iter()
            .map(|vg| {
                let start_gene = vg.cdr3_pos.ok_or(anyhow!("Model not fully loaded yet."))?;
                let index = self
                    .seg_vs
                    .iter()
                    .position(|x| x.name == vg.name)
                    .ok_or(anyhow!("Invalid V gene."))?;
                let pal_v = vg
                    .seq_with_pal
                    .as_ref()
                    .ok_or(anyhow!("Model not fully loaded yet."))?;
                let cdr3_pos = vg.cdr3_pos.ok_or(anyhow!("Model not fully loaded yet."))?;
                let start_seq = 0;
                let end_seq = pal_v.len() - cdr3_pos;
                let end_gene = start_gene + pal_v.len() - cdr3_pos;
                let mut errors = vec![0; self.p_del_v_given_v.dim().0];
                for del_v in 0..errors.len() {
                    if del_v <= pal_v.len() && del_v <= end_seq - start_seq {
                        errors[del_v] = count_differences(
                            &cdr3_seq.seq[0..end_seq - del_v],
                            &pal_v.seq[start_gene..end_gene - del_v],
                        );
                    }
                }

                Ok(VJAlignment {
                    index,
                    start_seq,
                    end_seq,
                    start_gene,
                    end_gene,
                    errors: errors,
                    score: 0, // meaningless
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let j_alignments = jgenes
            .iter()
            .map(|jg| {
                let index = self
                    .seg_js
                    .iter()
                    .position(|x| x.name == jg.name)
                    .ok_or(anyhow!("Invalid J gene."))?;
                let pal_j = jg
                    .seq_with_pal
                    .as_ref()
                    .ok_or(anyhow!("Model not fully loaded yet."))?;
                let cdr3_pos = jg.cdr3_pos.ok_or(anyhow!("Model not fully loaded yet."))?;
                let start_seq =
                    ((cdr3_seq.len() - cdr3_pos - 3) as i64 + self.range_del_j.0) as usize;
                let start_gene = 0;
                let end_seq = cdr3_seq.len();
                let end_gene = (cdr3_pos as i64 + 3 - self.range_del_j.0) as usize; // careful, palindromic insert
                let mut errors = vec![0; self.p_del_j_given_j.dim().0];
                for del_j in 0..errors.len() {
                    if del_j <= pal_j.len() && del_j <= end_gene - start_gene {
                        errors[del_j] = count_differences(
                            &cdr3_seq.seq[del_j + start_seq..end_seq],
                            &pal_j.seq[del_j + start_gene..end_gene],
                        );
                    }
                }

                Ok(VJAlignment {
                    index,
                    start_seq,
                    end_seq,
                    start_gene,
                    end_gene,
                    errors: errors,
                    score: 0, // meaningless
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut seq = Sequence {
            sequence: cdr3_seq.clone(),
            v_genes: v_alignments,
            j_genes: j_alignments,
            d_genes: Vec::new(),
            valid_alignment: true,
        };

        let align_params = AlignmentParameters::default();

        seq.d_genes = self.make_d_genes_alignments(&seq, &align_params)?;
        Ok(seq)
    }

    fn make_d_genes_alignments(
        &self,
        seq: &Sequence,
        align_params: &AlignmentParameters,
    ) -> Result<Vec<DAlignment>> {
        // roughly estimate bounds for the position of d
        // TODO: not great, improve on that
        let left_bound = seq
            .v_genes
            .iter()
            .map(|v| {
                if v.end_seq > self.p_del_v_given_v.dim().0 + self.p_del_d5_del_d3.dim().0 {
                    v.end_seq - (self.p_del_v_given_v.dim().0 + self.p_del_d5_del_d3.dim().0)
                } else {
                    0
                }
            })
            .min()
            .ok_or(anyhow!("Error in the definition of the D gene bounds"))?;

        let right_bound = seq
            .j_genes
            .iter()
            .map(|j| {
                cmp::min(
                    j.start_seq + (self.p_del_j_given_j.dim().0 + self.p_del_d5_del_d3.dim().1),
                    seq.sequence.len(),
                )
            })
            .max()
            .ok_or(anyhow!("Error in the definition of the D gene bounds"))?;

        // initialize all the d genes positions
        Ok(align_all_dgenes(
            &seq.sequence,
            self,
            left_bound,
            right_bound,
            align_params,
        ))
    }

    pub fn align_sequence(
        &self,
        dna_seq: Dna,
        align_params: &AlignmentParameters,
    ) -> Result<Sequence> {
        let mut seq = Sequence {
            sequence: dna_seq.clone(),
            v_genes: align_all_vgenes(&dna_seq, self, align_params),
            j_genes: align_all_jgenes(&dna_seq, self, align_params),
            d_genes: Vec::new(),
            valid_alignment: true,
        };

        // if we don't have v genes or j genes, don't try inferring the d gene
        if (seq.v_genes.is_empty()) | (seq.j_genes.is_empty()) {
            seq.valid_alignment = false;
            return Ok(seq);
        }
        seq.d_genes = self.make_d_genes_alignments(&seq, align_params)?;

        Ok(seq)
    }

    /// Re-create the full sequence of the variable region (with complete V/J gene, not just the CDR3)
    /// Return full_seq
    pub fn recreate_full_sequence(&self, dna: &Dna, vgene: &Gene, jgene: &Gene) -> Dna {
        let mut seq: Dna = Dna::new();
        let vgene_sans_cdr3 = vgene.seq.extract_subsequence(0, vgene.cdr3_pos.unwrap());
        seq.extend(&vgene_sans_cdr3);
        seq.extend(dna);
        seq.extend(
            &jgene
                .seq
                .extract_subsequence(jgene.cdr3_pos.unwrap() + 1, jgene.seq.len()),
        );
        seq
    }

    pub fn load_features(&mut self, feature: &Features) -> Result<()> {
        self.p_vdj = feature.vdj.probas.clone();
        self.p_del_v_given_v = feature.delv.probas.clone();
        self.set_p_vdj(&feature.vdj.probas.clone())?;
        self.p_del_j_given_j = feature.delj.probas.clone();
        self.p_del_d5_del_d3 = feature.deld.probas.clone();
        (self.p_ins_vd, self.markov_coefficients_vd) = feature.insvd.get_parameters();
        (self.p_ins_dj, self.markov_coefficients_dj) = feature.insdj.get_parameters();
        self.error_rate = feature.error.error_rate;

        Ok(())
    }

    pub fn update(&mut self, feature: &Features) -> Result<()> {
        self.load_features(feature)?;
        self.initialize()?;
        Ok(())
    }

    pub fn from_features(&self, feature: &Features) -> Result<Model> {
        let mut m = self.clone();
        m.update(feature)?;
        Ok(m)
    }

    pub fn set_p_vdj(&mut self, p_vdj: &Array3<f64>) -> Result<()> {
        // P(V,D,J) = P(D | V, J) * P(V, J) = P(D|V,J) * P(J|V)*P(V)
        self.p_vdj = p_vdj.clone();
        self.p_d_given_vj = Array3::zeros((p_vdj.dim().1, p_vdj.dim().0, p_vdj.dim().2));
        self.p_j_given_v = Array2::zeros((p_vdj.dim().2, p_vdj.dim().0));
        self.p_dj = Array2::zeros((p_vdj.dim().1, p_vdj.dim().2));
        self.p_v = Array1::zeros(p_vdj.dim().0);
        for vv in 0..p_vdj.dim().0 {
            for jj in 0..p_vdj.dim().2 {
                for dd in 0..p_vdj.dim().1 {
                    self.p_j_given_v[[jj, vv]] += self.p_vdj[[vv, dd, jj]];
                    self.p_d_given_vj[[dd, vv, jj]] += self.p_vdj[[vv, dd, jj]];
                    self.p_dj[[dd, jj]] += self.p_vdj[[vv, dd, jj]];
                    self.p_v[[vv]] += self.p_vdj[[vv, dd, jj]];
                }
            }
        }
        self.p_d_given_vj = self.p_d_given_vj.normalize_distribution()?;
        self.p_j_given_v = self.p_j_given_v.normalize_distribution()?;
        Ok(())
    }

    /// Check if the model is nearly identical to another model
    /// relative precision of 1e-4 to allow for numerical errors
    pub fn similar_to(&self, m: Model) -> bool {
        (self.seg_vs == m.seg_vs)
            && (self.seg_js == m.seg_js)
            && (self.seg_ds == m.seg_ds)
            && (self.seg_vs_sanitized == m.seg_vs_sanitized)
            && (self.seg_js_sanitized == m.seg_js_sanitized)
            && (self.p_d_given_vj.relative_eq(&m.p_d_given_vj, 1e-4, 1e-4))
            && self.p_v.relative_eq(&m.p_v, 1e-4, 1e-4)
            && self.p_ins_dj.relative_eq(&m.p_ins_dj, 1e-4, 1e-4)
            && self
                .p_del_v_given_v
                .relative_eq(&m.p_del_v_given_v, 1e-4, 1e-4)
            && self
                .p_del_j_given_j
                .relative_eq(&m.p_del_j_given_j, 1e-4, 1e-4)
            && self
                .p_del_d5_del_d3
                .relative_eq(&m.p_del_d5_del_d3, 1e-4, 1e-4)
            && self
                .markov_coefficients_vd
                .relative_eq(&m.markov_coefficients_vd, 1e-4, 1e-4)
            && self
                .markov_coefficients_dj
                .relative_eq(&m.markov_coefficients_dj, 1e-4, 1e-4)
            && (self.range_del_v == m.range_del_v)
            && (self.range_del_j == m.range_del_j)
            && (self.range_del_d3 == m.range_del_d3)
            && (self.range_del_d5 == m.range_del_d5)
            && ((self.error_rate - m.error_rate).abs() < 1e-4)
            && (self.thymic_q == m.thymic_q)
            && self.p_dj.relative_eq(&m.p_dj, 1e-4, 1e-4)
            && self.p_vdj.relative_eq(&m.p_vdj, 1e-4, 1e-4)
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
