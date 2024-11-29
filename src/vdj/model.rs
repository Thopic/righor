use crate::shared::gene::Gene;
use crate::shared::model::{sanitize_j, sanitize_v};
use crate::shared::parser::{
    parse_file, parse_str, EventType, Marginal, ParserMarginals, ParserParams,
};
use crate::shared::sequence::Dna;
use crate::shared::utils::send_warning;
use crate::shared::utils::Normalize2;
use crate::shared::{
    self,
    distributions::{calc_steady_state_dist, DiscreteDistribution, MarkovDNA},
    model::GenerationResult,
};
use crate::shared::{
    utils::sorted_and_complete, utils::sorted_and_complete_0start, utils::Normalize,
    utils::Normalize3, AlignmentParameters, AminoAcid, DAlignment, Features, InfEvent,
    InferenceParameters, ModelGen, ModelStructure, RecordModel, ResultInference, VJAlignment,
};
use crate::shared::{DNAMarkovChain, ErrorParameters, Modelable};
use crate::vdj::Features as FeaturesVDJ;

use crate::shared::sequence::SequenceType;
use crate::vdj::sequence::{align_all_dgenes, align_all_jgenes, align_all_vgenes};
use crate::vdj::{event::StaticEvent, Sequence};
use anyhow::{anyhow, Result};
use ndarray::{s, Array1, Array2, Array3, Axis};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

use kdam::TqdmParallelIterator;

use crate::{v_dj, vdj};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

use crate::shared::sequence::DnaLike;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use std::{cmp, fs::read_to_string, fs::File, io::Write};

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct Generator {
    model: Model,
    rng: SmallRng,
}

#[derive(Clone, Debug)]
/// Make infer a generic function by allowing different entry
pub enum EntrySequence {
    Aligned(Sequence),
    NucleotideSequence(DnaLike),
    NucleotideCDR3((DnaLike, Vec<Gene>, Vec<Gene>)),
}

impl EntrySequence {
    pub fn align(&self, model: &Model, align_params: &AlignmentParameters) -> Result<Sequence> {
        match self {
            EntrySequence::Aligned(x) => Ok(x.clone()),
            EntrySequence::NucleotideSequence(seq) => {
                model.align_sequence(seq.clone(), align_params)
            }
            EntrySequence::NucleotideCDR3((seq, v, j)) => model.align_from_cdr3(seq, v, j),
        }
    }

    pub fn compatible_with_inference(&self) -> bool {
        match self {
            EntrySequence::Aligned(s) => !s.sequence.is_ambiguous(),
            EntrySequence::NucleotideSequence(s) => !s.is_ambiguous(),
            EntrySequence::NucleotideCDR3((s, _, _)) => !s.is_ambiguous(),
        }
    }

    pub fn is_protein(&self) -> bool {
        match self {
            EntrySequence::Aligned(s) => s.sequence.is_protein(),
            EntrySequence::NucleotideSequence(s) => s.is_protein(),
            EntrySequence::NucleotideCDR3((s, _, _)) => s.is_protein(),
        }
    }
}

impl Generator {
    /// `available_v`, `available_j`: set of v,j genes to choose from
    pub fn new(
        model: &Model,
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

        if available_v.is_some() {
            internal_model = internal_model.filter_vs(available_v.unwrap())?;
        }
        if available_j.is_some() {
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
    pub fn generate(&mut self, functional: bool) -> Result<GenerationResult> {
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
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    // Schema of the model
    pub model_type: ModelStructure,

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
    #[serde(skip)]
    pub gen: Generative,
    //    pub markov_coefficients_vd: Array2<f64>,
    pub markov_chain_vd: Arc<DNAMarkovChain>,
    pub markov_chain_dj: Arc<DNAMarkovChain>,
    //   pub markov_coefficients_dj: Array2<f64>,
    pub range_del_v: (i64, i64), // range of "real deletions", e.g -5, 12 (neg. del. => pal. ins.)
    pub range_del_j: (i64, i64),
    pub range_del_d3: (i64, i64),
    pub range_del_d5: (i64, i64),
    pub error: ErrorParameters,

    // Not directly useful for the model but useful for integration with other soft
    // TODO: transform these in "getter" in the python bindings, they don't need to be precomputed
    pub p_v: Array1<f64>,
    pub p_dj: Array2<f64>,
    pub p_d_given_vj: Array3<f64>,
    pub p_j_given_v: Array2<f64>,
    pub thymic_q: f64,
}

impl Modelable for Model {
    fn load_from_name(
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

    fn load_from_files(
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

    fn load_from_str(
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
    fn save_model(&self, directory: &Path) -> Result<()> {
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

    /// Save the data in json format
    fn save_json(&self, filename: &Path) -> Result<()> {
        let mut file = File::create(filename)?;
        let json = serde_json::to_string(&self)?;
        Ok(writeln!(file, "{json}")?)
    }

    /// Load a saved model in json format
    fn load_json(filename: &Path) -> Result<Model> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let mut model: Model = serde_json::from_reader(reader)?;
        model.initialize()?;
        Ok(model)
    }

    /// Return (`cdr3_nt`, `cdr3_aa`, `full_sequence`, `event`, `vname`, `jname`)
    fn generate<R: Rng>(&mut self, functional: bool, rng: &mut R) -> Result<GenerationResult> {
        let (mut full_seq, _, _, mut event) = self.generate_no_error(functional, rng);

        let mut generic_event = shared::StaticEvent::VDJ(event);
        // add errors
        self.error
            .apply_to_sequence(&full_seq, &mut generic_event, rng);
        event = match generic_event.clone() {
            shared::StaticEvent::VDJ(x) => x,
            _ => unreachable!(),
        };
        full_seq = event.to_sequence(self);

        let cdr3_nt = event.extract_cdr3(&full_seq, self);
        let cdr3_aa = cdr3_nt.translate().ok();

        Ok(GenerationResult {
            v_gene: self.seg_vs[event.v_index].name.clone(),
            j_gene: self.seg_js[event.j_index].name.clone(),
            recombination_event: generic_event,
            full_seq: full_seq.get_string(),
            junction_nt: cdr3_nt.get_string(),
            junction_aa: cdr3_aa.map(|x| x.to_string()),
        })
    }

    /// Return (`cdr3_nt`, `cdr3_aa`, `full_sequence`, `event`, `vname`, `jname`)
    fn generate_without_errors<R: Rng>(
        &mut self,
        functional: bool,
        rng: &mut R,
    ) -> GenerationResult {
        let (full_seq, cdr3_nt, cdr3_aa, event) = self.generate_no_error(functional, rng);
        let generic_event = shared::StaticEvent::VDJ(event.clone());
        GenerationResult {
            v_gene: self.seg_vs[event.v_index].name.clone(),
            j_gene: self.seg_js[event.j_index].name.clone(),
            recombination_event: generic_event,
            full_seq: full_seq.get_string(),
            junction_nt: cdr3_nt.get_string(),
            junction_aa: cdr3_aa.map(|x| x.to_string()),
        }
    }

    fn uniform(&self) -> Result<Model> {
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
            markov_chain_vd: Arc::new(DNAMarkovChain::new(
                &Array2::<f64>::ones(self.markov_chain_vd.transition_matrix.dim()),
                false,
            )?),
            markov_chain_dj: Arc::new(DNAMarkovChain::new(
                &Array2::<f64>::ones(self.markov_chain_dj.transition_matrix.dim()),
                true, // reversed
            )?),

            //            markov_coefficients_vd: Array2::<f64>::ones(self.markov_coefficients_vd.dim()),
            // markov_coefficients_dj: Array2::<f64>::ones(self.markov_coefficients_dj.dim()),
            error: ErrorParameters::uniform(&self.error)?,
            model_type: self.model_type.clone(),
            ..Default::default()
        };
        m.initialize()?;
        Ok(m)
    }

    /// Re-initialize the error model, normalize the parameters
    fn initialize(&mut self) -> Result<()> {
        self.sanitize_genes()?;

        self.p_vdj = self.p_vdj.normalize_distribution_3()?;
        self.set_p_vdj(&self.p_vdj.clone())?;
        self.p_ins_vd = self.p_ins_vd.normalize_distribution()?;
        self.p_ins_dj = self.p_ins_dj.normalize_distribution()?;
        self.p_del_v_given_v = self.p_del_v_given_v.normalize_distribution()?;
        self.p_del_j_given_j = self.p_del_j_given_j.normalize_distribution()?;
        self.p_del_d5_del_d3 = self.p_del_d5_del_d3.normalize_distribution_double()?;
        // self.markov_coefficients_vd = self.markov_coefficients_vd.normalize_last()?;
        // self.markov_coefficients_dj = self.markov_coefficients_vd.normalize_last()?;

        self.initialize_generative_model()?;
        self.safety_checks();
        Ok(())
    }

    /// One step of expectation maximization given a set of sequences
    fn infer(
        &mut self,
        sequences: &[EntrySequence],
        features_opt: Option<Vec<Features>>,
        alignment_params: &AlignmentParameters,
        inference_params: &InferenceParameters,
    ) -> Result<(Vec<Features>, f64)> {
        if !sequences
            .iter()
            .all(EntrySequence::compatible_with_inference)
        {
            return Err(anyhow!(
                "Cannot do inference when sequences have ambiguity. \
				Ambiguous nucleotides (N) or protein sequence \
				are out."
            ));
        }

        let mut ip = inference_params.clone();

        // no need to compute pgen or store best event if we're infering
        ip.compute_pgen = false;
        ip.store_best_event = false;

        let features = match features_opt {
            None => {
                // Create new features object
                vec![
                    match self.model_type {
                        ModelStructure::VDJ => Features::VDJ(vdj::Features::new(self)?),
                        ModelStructure::VxDJ => Features::VxDJ(v_dj::Features::new(self)?),
                    };
                    sequences.len()
                ]
            }
            Some(feats) => feats,
        };

        let new_features = (&features, sequences)
            .into_par_iter()
            .tqdm()
            .map(|(feat, sequence)| {
                let aligned = sequence.align(self, alignment_params)?;
                let mut new_feat = feat.clone();
                let _ = new_feat.infer(&aligned, &ip)?;
                Ok(new_feat)
            })
            .collect::<Result<Vec<_>>>()?;

        // update the model and clean up the features
        Features::update(new_features, self, &ip)
    }

    /// Evaluate a sequence and return the result of the inference
    fn evaluate(
        &self,
        sequence: EntrySequence,
        alignment_params: &AlignmentParameters,
        inference_params: &InferenceParameters,
    ) -> Result<ResultInference> {
        let mut ip = inference_params.clone();
        if sequence.is_protein() {
            ip.do_not_infer_features();
        }

        let mut features = match self.model_type {
            ModelStructure::VDJ => Features::VDJ(vdj::Features::new(self)?),
            ModelStructure::VxDJ => Features::VxDJ(v_dj::Features::new(self)?),
        };

        let aligned_sequence = sequence.align(self, alignment_params)?;
        let mut result = features.infer(&aligned_sequence, &ip)?;
        result.fill_event(self, &aligned_sequence)?;

        // no error: likelihood = pgen
        if self.error.no_error() {
            result.pgen = result.likelihood;
            return Ok(result);
        }

        // likelihood is 0, so pgen is also 0
        if result.likelihood == 0. {
            result.pgen = 0.;
            return Ok(result);
        }

        // Otherwise, we need to compute the pgen of the reconstructed sequence
        if ip.compute_pgen && ip.store_best_event {
            let event = result
                .get_best_event()
                .ok_or(anyhow!("Error with event extraction during pgen inference"))?;
            let cdr3_nt = event.clone().get_reconstructed_cdr3(self)?;
            let cdr3: DnaLike = match aligned_sequence.sequence_type {
                SequenceType::Dna => cdr3_nt.into(),
                SequenceType::Protein => cdr3_nt.translate()?.into(),
            };

            // let seq_without_err = event.reconstructed_sequence.ok_or(anyhow!(
            //     "Error with event reconstruction during pgen inference"
            // ))?;

            // full sequence, so default alignment parameters should be fine.
            // except it's way too slow. Just do the CDR3 instead.
            let aligned_seq = self.align_from_cdr3(
                &cdr3,
                &[self.seg_vs[event.v_index].clone()],
                &[self.seg_js[event.j_index].clone()],
            )?;

            let mut features_pgen = match self.model_type {
                ModelStructure::VDJ => Features::VDJ(vdj::Features::new(self)?),
                ModelStructure::VxDJ => Features::VxDJ(v_dj::Features::new(self)?),
            };
            features_pgen.error_mut().remove_error()?; // remove the error

            result.pgen = features_pgen.infer(&aligned_seq, &ip)?.likelihood;
        }
        Ok(result)
    }

    fn filter_vs(&self, vs: Vec<Gene>) -> Result<Model> {
        let mut m = self.clone();

        let dim = self.p_vdj.dim();
        let mut p_vdj = Array3::<f64>::zeros((vs.len(), dim.1, dim.2));
        m.seg_vs = Vec::new();
        m.p_del_v_given_v = Array2::<f64>::zeros((self.p_del_v_given_v.dim().0, vs.len()));

        let mut iv_restr = 0;
        for iv in 0..dim.0 {
            let vgene = self.seg_vs[iv].clone();
            if vs.contains(&vgene) {
                m.seg_vs.push(vgene);
                for id in 0..dim.1 {
                    for ij in 0..dim.2 {
                        p_vdj[[iv_restr, id, ij]] = self.p_vdj[[iv, id, ij]];
                    }
                }
                for idelv in 0..self.p_del_v_given_v.dim().0 {
                    m.p_del_v_given_v[[idelv, iv_restr]] = self.p_del_v_given_v[[idelv, iv]];
                }
                iv_restr += 1;
            }
        }

        p_vdj = p_vdj.normalize_distribution_3()?;
        m.set_p_vdj(&p_vdj.clone())?;
        m.initialize()?;
        Ok(m)
    }

    fn filter_js(&self, js: Vec<Gene>) -> Result<Model> {
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

    fn align_from_cdr3(
        &self,
        cdr3_seq: &DnaLike,
        vgenes: &[Gene],
        jgenes: &[Gene],
    ) -> Result<Sequence> {
        let mut v_alignments = vgenes
            .iter()
            .map(|vg| {
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
                if pal_v.len() < cdr3_pos {
                    return Err(anyhow!(
                        "cdr3 position farther up than v length ({})",
                        vg.name
                    ));
                }
                let start_gene = cdr3_pos;
                let end_seq = pal_v.len() - cdr3_pos;
                let end_gene = start_gene + pal_v.len() - cdr3_pos;

                let mut val = VJAlignment {
                    index,
                    start_seq,
                    end_seq,
                    start_gene,
                    end_gene,
                    score: 0, // meaningless
                    max_del: Some(self.p_del_v_given_v.shape()[0]),
                    gene_sequence: pal_v.clone(),
                    sequence_type: cdr3_seq.sequence_type(),
                    ..Default::default()
                };
                val.precompute_errors_v(cdr3_seq);
                Ok(val)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut j_alignments = jgenes
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

                if cdr3_pos + 3 > pal_j.len() {
                    return Err(anyhow!(
                        "cdr3 position farther up than j length ({})",
                        jg.name
                    ));
                }

                let end_seq = cdr3_seq.len();
                let end_gene = ((cdr3_pos as i64) - self.range_del_j.0 + 3) as usize;
                let start_gene =
                    if (cdr3_pos as i64) - self.range_del_j.0 <= cdr3_seq.len() as i64 - 3 {
                        0
                    } else {
                        cdr3_pos as i64 - self.range_del_j.0 - cdr3_seq.len() as i64 + 3
                    } as usize;

                let start_seq =
                    if (cdr3_pos as i64) - self.range_del_j.0 <= cdr3_seq.len() as i64 - 3 {
                        cdr3_seq.len() as i64 - end_gene as i64
                    } else {
                        0
                    } as usize;

                debug_assert!(end_seq - start_seq > 0); // should be fine

                let mut jal = VJAlignment {
                    index,
                    start_seq,
                    end_seq,
                    start_gene,
                    end_gene,
                    score: 0, // meaningless
                    max_del: Some(self.p_del_j_given_j.dim().0),
                    gene_sequence: pal_j.clone(),
                    sequence_type: cdr3_seq.sequence_type(),
                    ..Default::default()
                };
                jal.precompute_errors_j(cdr3_seq);
                Ok(jal)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut sequence = cdr3_seq.clone();
        if v_alignments.len() == 1 {
            // if only one V, we extend the seq
            let mut only_val = v_alignments[0].clone();
            if only_val.start_seq != 0 {
                // weird edge case, the sequence starts before the V gene
                // shouldn't happend, but there's a debug_assert just in case
                debug_assert!(only_val.start_seq == 0);
            } else {
                let vg: DnaLike = only_val
                    .gene_sequence
                    .extract_subsequence(0, only_val.start_gene)
                    .into();
                only_val.start_seq = 0;
                only_val.end_seq += only_val.start_gene;
                sequence = vg.extended_in_frame(&sequence);
                for jal in &mut j_alignments {
                    jal.start_seq += only_val.start_gene;
                    jal.end_seq += only_val.start_gene;
                }
                // need to do this last
                only_val.start_gene = 0;
                v_alignments = vec![only_val];
            }
        }
        if j_alignments.len() == 1 {
            // same if only one J
            let mut only_jal = j_alignments[0].clone();
            // maybe a bit more common, if the J gene end before the CDR3
            // we don't do anything
            if only_jal.start_seq == 0 {
                let jg: DnaLike = only_jal
                    .gene_sequence
                    .extract_subsequence(only_jal.end_gene, only_jal.gene_sequence.len())
                    .into();
                sequence = sequence.extended_in_frame(&jg);
                only_jal.end_seq += only_jal.gene_sequence.len() - only_jal.end_gene;
                // need to do this last
                only_jal.end_gene = only_jal.gene_sequence.len();
                j_alignments = vec![only_jal];
            }
        }

        let mut seq = Sequence {
            sequence,
            v_genes: v_alignments,
            j_genes: j_alignments,
            d_genes: Vec::new(),
            valid_alignment: true,
            sequence_type: cdr3_seq.sequence_type(),
        };

        let align_params = AlignmentParameters::default();

        seq.d_genes = self.make_d_genes_alignments(&seq, &align_params)?;

        Ok(seq)
    }

    fn align_sequence(
        &self,
        dna_seq: DnaLike,
        align_params: &AlignmentParameters,
    ) -> Result<Sequence> {
        let mut seq = Sequence {
            sequence: dna_seq.clone(),
            v_genes: align_all_vgenes(&dna_seq.clone(), self, align_params),
            j_genes: align_all_jgenes(&dna_seq.clone(), self, align_params),
            d_genes: Vec::new(),
            valid_alignment: true,
            sequence_type: dna_seq.sequence_type(),
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
    /// Return `full_seq`
    fn recreate_full_sequence(&self, dna_cdr3: &Dna, vgene: &Gene, jgene: &Gene) -> Dna {
        let mut seq: Dna = Dna::new();
        let vgene_sans_cdr3 = vgene.seq.extract_subsequence(0, vgene.cdr3_pos.unwrap());
        seq.extend(&vgene_sans_cdr3);
        seq.extend(dna_cdr3);
        seq.extend(
            &jgene
                .seq
                .extract_subsequence(jgene.cdr3_pos.unwrap() + 1, jgene.seq.len()),
        );
        seq
    }

    /// Check if the model is nearly identical to another model
    /// relative precision of 1e-4 to allow for numerical errors
    fn similar_to(&self, m: Model) -> bool {
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
            && self.markov_chain_vd.transition_matrix.relative_eq(
                &m.markov_chain_vd.transition_matrix,
                1e-4,
                1e-4,
            )
            && self.markov_chain_dj.transition_matrix.relative_eq(
                &m.markov_chain_dj.transition_matrix,
                1e-4,
                1e-4,
            )
            && (self.range_del_v == m.range_del_v)
            && (self.range_del_j == m.range_del_j)
            && (self.range_del_d3 == m.range_del_d3)
            && (self.range_del_d5 == m.range_del_d5)
            && ErrorParameters::similar(self.error.clone(), m.error)
            && ((self.thymic_q - m.thymic_q).abs() < 1e-40)
            && self.p_dj.relative_eq(&m.p_dj, 1e-4, 1e-4)
            && self.p_vdj.relative_eq(&m.p_vdj, 1e-4, 1e-4)
    }
}

impl Model {
    pub fn infer_brute_force(
        &mut self,
        sequences: &[EntrySequence],
        features_opt: Option<Vec<Features>>,
        alignment_params: &AlignmentParameters,
        inference_params: &InferenceParameters,
    ) -> Result<(Vec<Features>, f64)> {
        if sequences.iter().any(EntrySequence::is_protein) {
            return Err(anyhow!("The brute-force model doesn't work with proteins"));
        }

        let features = match features_opt {
            None => {
                // Create new features object
                vec![
                    match self.model_type {
                        ModelStructure::VDJ => Features::VDJ(vdj::Features::new(self)?),
                        ModelStructure::VxDJ => Features::VxDJ(v_dj::Features::new(self)?),
                    };
                    sequences.len()
                ]
            }
            Some(feats) => feats,
        };

        let new_features = (&features, sequences)
            .into_par_iter()
            .map(|(feat, sequence)| {
                let aligned = sequence.align(self, alignment_params)?;
                let new_feat = feat.clone();
                let mut feat_vdj = match new_feat {
                    Features::VDJ(x) => Ok(x),
                    Features::VxDJ(_) => Err(anyhow!("Shouldn't happen.")),
                }?;

                let _ = feat_vdj.infer_brute_force(&aligned, inference_params)?;
                Ok(Features::VDJ(feat_vdj))
            })
            .collect::<Result<Vec<_>>>()?;
        Features::update(new_features, self, inference_params)
    }

    pub fn evaluate_brute_force(
        &self,
        sequence: &EntrySequence,
        alignment_params: &AlignmentParameters,
        inference_params: &InferenceParameters,
    ) -> Result<ResultInference> {
        let mut feature = FeaturesVDJ::new(self)?;
        let aligned = sequence.align(self, alignment_params)?;
        let mut result = feature.infer_brute_force(&aligned, inference_params)?;
        result.fill_event(self, &aligned)?;
        Ok(result)
    }

    pub fn get_p_vj(&self) -> Array2<f64> {
        self.p_vdj.sum_axis(Axis(1))
    }

    pub fn get_p_d_given_j(&self) -> Array2<f64> {
        let pdj = self.p_vdj.sum_axis(Axis(0));
        let pj = pdj.sum_axis(Axis(0)).insert_axis(Axis(0));
        (pdj / pj).mapv(|x| if x.is_nan() { 0.0 } else { x })
    }

    pub fn write_v_anchors(&self) -> Result<String> {
        let mut wtr = csv::Writer::from_writer(vec![]);
        wtr.write_record(["gene", "anchor_index", "function"])?;
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
        wtr.write_record(["gene", "anchor_index", "function"])?;
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
            / p_deld5_given_d.broadcast(p_d3_d5_d.clone().dim()).unwrap())
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
            self.markov_chain_vd
                .transition_matrix
                .iter()
                .copied()
                .collect::<Array1<f64>>()
                .into_dyn(),
        )
        .write()?;
        let marginal_djins =
            Marginal::create(Vec::new(), self.p_ins_dj.clone().into_dyn()).write()?;
        let marginal_djdinucl = Marginal::create(
            Vec::new(),
            self.markov_chain_dj
                .transition_matrix
                .iter()
                .copied()
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
        let delvs = EventType::Numbers((self.range_del_v.0..=self.range_del_v.1).collect());
        result.push_str(&delvs.write());

        result.push_str("#Deletion;D_gene;Three_prime;5;d_3_del\n");
        let deld3s = EventType::Numbers((self.range_del_d3.0..=self.range_del_d3.1).collect());
        result.push_str(&deld3s.write());

        result.push_str("#Deletion;D_gene;Five_prime;5;d_5_del\n");
        let deld5s = EventType::Numbers((self.range_del_d5.0..=self.range_del_d5.1).collect());
        result.push_str(&deld5s.write());

        result.push_str("#Deletion;J_gene;Five_prime;5;j_5_del\n");
        let deljs = EventType::Numbers((self.range_del_j.0..=self.range_del_j.1).collect());
        result.push_str(&deljs.write());

        result.push_str("#Insertion;VD_genes;Undefined_side;4;vd_ins\n");
        let insvds = EventType::Numbers((0_i64..self.p_ins_vd.dim() as i64).collect());
        result.push_str(&insvds.write());

        result.push_str("#Insertion;DJ_genes;Undefined_side;2;dj_ins\n");
        let insdjs = EventType::Numbers((0_i64..self.p_ins_dj.dim() as i64).collect());
        result.push_str(&insdjs.write());

        let dimv = self.seg_vs.len();
        let dimdelv = self.p_del_v_given_v.dim().0;
        let dimd = self.seg_ds.len();
        let dimdeld3 = self.p_del_d5_del_d3.dim().1;
        let dimdeld5 = self.p_del_d5_del_d3.dim().0;
        let dimj = self.seg_js.len();
        let dimdelj = self.p_del_j_given_j.dim().0;
        let error = self.error.write();
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
	     {error}"
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

        if !(sorted_and_complete(&arrdelv)
            & sorted_and_complete(&arrdelj)
            & sorted_and_complete(&arrdeld3)
            & sorted_and_complete(&arrdeld5)
            & sorted_and_complete_0start(
                &pp.params
                    .get("vd_ins")
                    .ok_or(anyhow!("Invalid vd_ins"))?
                    .clone()
                    .to_numbers()?,
            )
            & sorted_and_complete_0start(
                &pp.params
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
            return Err(anyhow!("Wrong format for the VDJ probabilities"));
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
                    model.p_del_d5_del_d3[[d5, d3, dd]] = pdeld3[[dd, d5, d3]] * pdeld5[[dd, d5]];
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
        model.markov_chain_vd = Arc::new(DNAMarkovChain::new(
            &pm.marginals
                .get("vd_dinucl")
                .unwrap()
                .probabilities
                .clone()
                .into_shape((4, 4))
                .map_err(|_e| anyhow!("Wrong size for vd_dinucl"))?,
            false,
        )?);
        model.markov_chain_dj = Arc::new(DNAMarkovChain::new(
            &pm.marginals
                .get("dj_dinucl")
                .unwrap()
                .probabilities
                .clone()
                .into_shape((4, 4))
                .map_err(|_e| anyhow!("Wrong size for dj_dinucl"))?,
            true,
        )?);

        // TODO: Need to deal with potential first nt bias in the file
        // model.first_nt_bias_ins_vd =
        //     Array1::from_vec(calc_steady_state_dist(&model.markov_coefficients_vd)?);
        // model.first_nt_bias_ins_dj =
        //     Array1::from_vec(calc_steady_state_dist(&model.markov_coefficients_dj)?);

        model.error = pp.error.clone();
        model.thymic_q = 9.41; // TODO: deal with this

        model.initialize()?;
        Ok(model)
    }

    /// Emit a warning if stuff are weird
    /// Right now not much, but check the functionality of the genes
    pub fn safety_checks(&self) {
        if self.seg_vs.iter().all(|x| !x.is_functional) {
            send_warning("All the V genes in the model are tagged as non-functional. This could result in an infinite loop if trying to generate functional sequences.\n");
        }
        if self.seg_js.iter().all(|x| !x.is_functional) {
            send_warning("All the J genes in the model are tagged as non-functional. This could result in an infinite loop if trying to generate functional sequences.\n");
        }
    }

    pub fn sanitize_genes(&mut self) -> Result<()> {
        // Trim the V/J nucleotides sequences at the CDR3 region (include F/W/C residues)
        // and append the maximum number of reverse palindromic insertions appended.

        // Add the palindromic insertions
        for g in &mut self.seg_vs {
            g.create_palindromic_ends(0, (-self.range_del_v.0) as usize);
        }
        for g in &mut self.seg_js {
            g.create_palindromic_ends((-self.range_del_j.0) as usize, 0);
        }

        for g in &mut self.seg_ds {
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
        self.gen.d_vdj =
            DiscreteDistribution::new(&self.p_vdj.view().iter().copied().collect::<Vec<_>>())?;
        self.gen.d_ins_vd = DiscreteDistribution::new(&self.p_ins_vd.to_vec())?;
        self.gen.d_ins_dj = DiscreteDistribution::new(&self.p_ins_dj.to_vec())?;

        self.gen.d_del_v_given_v = Vec::new();
        for row in self.p_del_v_given_v.axis_iter(Axis(1)) {
            self.gen
                .d_del_v_given_v
                .push(DiscreteDistribution::new(&row.to_vec())?);
        }
        self.gen.d_del_j_given_j = Vec::new();
        for row in self.p_del_j_given_j.axis_iter(Axis(1)) {
            self.gen
                .d_del_j_given_j
                .push(DiscreteDistribution::new(&row.to_vec())?);
        }

        self.gen.d_del_d5_del_d3 = Vec::new();
        for ddd in 0..self.p_del_d5_del_d3.dim().2 {
            let d5d3: Vec<f64> = self
                .p_del_d5_del_d3
                .slice(s![.., .., ddd])
                .iter()
                .copied()
                .collect();
            self.gen
                .d_del_d5_del_d3
                .push(DiscreteDistribution::new(&d5d3)?);
        }

        self.gen.markov_vd = MarkovDNA::new(&self.markov_chain_vd.transition_matrix.to_owned())?;
        self.gen.markov_dj = MarkovDNA::new(&self.markov_chain_dj.transition_matrix.to_owned())?;

        Ok(())
    }

    /// Update the v segments and adapt the associated marginals
    pub fn set_v_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        let [_, sd, sj] = *self.p_vdj.shape() else {
            return Err(anyhow!("Something is wrong with the v segments"));
        };
        let mut new_p_vdj = Array3::<f64>::zeros([value.len(), sd, sj]);

        let [sdelv, _] = *self.p_del_v_given_v.shape() else {
            return Err(anyhow!("Something is wrong with the v segments"));
        };
        let mut new_p_del_v_given_v = Array2::<f64>::zeros([sdelv, value.len()]);

        let proba_v_default = 1. / (value.len() as f64);
        let delv_default = self.p_del_v_given_v.sum_axis(Axis(1)) / self.p_del_v_given_v.sum();

        for (iv, v) in value.iter().enumerate() {
            match self
                .seg_vs
                .iter()
                .enumerate()
                .find(|(_index, g)| g.name == v.name)
            {
                Some((index, _gene)) => {
                    new_p_vdj
                        .slice_mut(s![iv, .., ..])
                        .assign(&self.p_vdj.slice_mut(s![index, .., ..]));
                    new_p_del_v_given_v
                        .slice_mut(s![.., iv])
                        .assign(&self.p_del_v_given_v.slice_mut(s![.., index]));
                }
                None => {
                    new_p_vdj.slice_mut(s![iv, .., ..]).fill(proba_v_default);
                    new_p_del_v_given_v
                        .slice_mut(s![.., iv])
                        .assign(&delv_default);
                }
            }
        }

        // normalzie
        new_p_vdj = new_p_vdj.normalize_distribution_3()?;
        new_p_del_v_given_v = new_p_del_v_given_v.normalize_distribution()?;

        self.seg_vs = value;
        self.set_p_vdj(&new_p_vdj)?;
        self.p_del_v_given_v = new_p_del_v_given_v;
        self.initialize()?;
        Ok(())
    }

    /// Update the j segments and adapt the associated marginals
    pub fn set_j_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        let [sv, sd, _] = *self.p_vdj.shape() else {
            return Err(anyhow!("Something is wrong with the j segments"));
        };
        let mut new_p_vdj = Array3::<f64>::zeros([sv, sd, value.len()]);

        let [sdelj, _] = *self.p_del_j_given_j.shape() else {
            return Err(anyhow!("Something is wrong with the j segments"));
        };
        let mut new_p_del_j_given_j = Array2::<f64>::zeros([sdelj, value.len()]);

        let proba_j_default = 1. / (value.len() as f64);
        let delj_default = self.p_del_j_given_j.sum_axis(Axis(1)) / self.p_del_j_given_j.sum();

        for (ij, j) in value.iter().enumerate() {
            match self
                .seg_js
                .iter()
                .enumerate()
                .find(|(_index, g)| g.name == j.name)
            {
                Some((index, _gene)) => {
                    new_p_vdj
                        .slice_mut(s![.., .., ij])
                        .assign(&self.p_vdj.slice_mut(s![.., .., index]));
                    new_p_del_j_given_j
                        .slice_mut(s![.., ij])
                        .assign(&self.p_del_j_given_j.slice_mut(s![.., index]));
                }
                None => {
                    new_p_vdj.slice_mut(s![.., .., ij]).fill(proba_j_default);
                    new_p_del_j_given_j
                        .slice_mut(s![.., ij])
                        .assign(&delj_default);
                }
            }
        }

        // normalzie
        new_p_vdj = new_p_vdj.normalize_distribution_3()?;
        new_p_del_j_given_j = new_p_del_j_given_j.normalize_distribution()?;

        self.seg_js = value;
        self.set_p_vdj(&new_p_vdj)?;
        self.p_del_j_given_j = new_p_del_j_given_j;
        self.initialize()?;
        Ok(())
    }

    /// Update the d segments and adapt the associated marginals
    /// The new d segments (the ones with a different name), get probability equal
    /// to `1/number_d_segments` and average deletion profiles.
    pub fn set_d_segments(&mut self, value: Vec<Gene>) -> Result<()> {
        let [sv, _, sj] = *self.p_vdj.shape() else {
            return Err(anyhow!("Something is wrong with the v segments"));
        };
        let mut new_p_vdj = Array3::<f64>::zeros([sv, value.len(), sj]);

        let [sdeld5, sdeld3, _] = *self.p_del_d5_del_d3.shape() else {
            return Err(anyhow!("Something is wrong with the v segments"));
        };
        let mut new_p_del_d5_del_d3 = Array3::<f64>::zeros([sdeld5, sdeld3, value.len()]);

        let proba_d_default = 1. / (value.len() as f64);
        let deld5_deld3_default =
            self.p_del_d5_del_d3.sum_axis(Axis(2)) / self.p_del_d5_del_d3.sum();

        for (id, d) in value.iter().enumerate() {
            // try to see if we find a gene with the same name,
            match self
                .seg_ds
                .iter()
                .enumerate()
                .find(|(_index, g)| g.name == d.name)
            {
                Some((index, _gene)) => {
                    new_p_vdj
                        .slice_mut(s![.., id, ..])
                        .assign(&self.p_vdj.slice_mut(s![.., index, ..]));
                    new_p_del_d5_del_d3
                        .slice_mut(s![.., .., id])
                        .assign(&self.p_del_d5_del_d3.slice_mut(s![.., .., index]));
                }
                None => {
                    new_p_vdj.slice_mut(s![.., id, ..]).fill(proba_d_default);
                    new_p_del_d5_del_d3
                        .slice_mut(s![.., .., id])
                        .assign(&deld5_deld3_default);
                }
            }
        }

        // normalize
        new_p_vdj = new_p_vdj.normalize_distribution_3()?;
        new_p_del_d5_del_d3 = new_p_del_d5_del_d3.normalize_distribution_double()?;

        self.seg_ds = value;
        self.set_p_vdj(&new_p_vdj)?;
        self.p_del_d5_del_d3 = new_p_del_d5_del_d3;
        self.initialize()?;
        Ok(())
    }

    /// Return (`full_seq`, `cdr3_seq`, `aa_seq`, `event`)
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
            // if the V gene is not functional and we're looking for functional seqs
            if functional && !self.seg_vs[event.v_index].is_functional() {
                continue;
            }
            event.j_index = vdj_index % self.p_dj.dim().1;
            // same for J gene
            if functional && !self.seg_js[event.j_index].is_functional() {
                continue;
            }

            event.d_index =
                (vdj_index % (self.p_vdj.dim().1 * self.p_vdj.dim().2)) / self.p_vdj.dim().2;

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
            event.d_start_seq = (seq_v.len() - event.delv - event.deld5 + ins_vd) as i64;
            event.j_start_seq = (seq_v.len() - event.delv + ins_vd + ins_dj + seq_d.len()
                - event.deld5
                - event.deld3
                - event.delj) as i64;

            // create the complete sequence:
            let full_seq = event.to_sequence(self);

            // create the complete CDR3 sequence
            let cdr3_seq = event.to_cdr3(self);

            // if the cdr3 is empty (too much cutting) we assume the resulting sequence
            // is not functional
            if functional && cdr3_seq.is_empty() {
                continue;
            }

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

                    if functional
                        && saa.seq.last().copied().unwrap() != b'F'
                        && saa.seq.last().copied().unwrap() != b'W'
                    {
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

    pub fn get_v_gene(&self, event: &InfEvent) -> String {
        self.seg_vs[event.v_index].name.clone()
    }

    pub fn get_j_gene(&self, event: &InfEvent) -> String {
        self.seg_js[event.j_index].name.clone()
    }

    pub fn get_d_gene(&self, event: &InfEvent) -> String {
        self.seg_ds[event.d_index].name.clone()
    }

    pub fn get_first_nt_bias_ins_vd(&self) -> Result<Vec<f64>> {
        calc_steady_state_dist(&self.markov_chain_vd.transition_matrix)
    }

    pub fn get_first_nt_bias_ins_dj(&self) -> Result<Vec<f64>> {
        calc_steady_state_dist(&self.markov_chain_dj.transition_matrix)
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
                v.end_seq as i64
                    - (self.p_del_v_given_v.dim().0 + self.p_del_d5_del_d3.dim().0) as i64
            })
            .min()
            .ok_or(anyhow!("Error in the definition of the D gene bounds"))?;

        let right_bound = seq
            .j_genes
            .iter()
            .map(|j| {
                cmp::min(
                    (j.start_seq as i64 - j.start_gene as i64)
                        + (self.p_del_j_given_j.dim().0 + self.p_del_d5_del_d3.dim().1) as i64,
                    seq.sequence.len() as i64,
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

    // pub fn update(&mut self, feature: &Features) -> Result<()> {
    //     feature.update_model(self)?;
    //     self.initialize()?;
    //     Ok(())
    // }

    // pub fn from_features(&self, feature: &Features) -> Result<Model> {
    //     let mut m = self.clone();
    //     m.update(feature)?;
    //     Ok(m)
    // }

    pub fn set_p_vdj(&mut self, p_vdj: &Array3<f64>) -> Result<()> {
        // P(V,D,J) = P(D | V, J) * P(V, J) = P(D|V,J) * P(J|V)*P(V)
        self.p_vdj.clone_from(p_vdj);
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
}

impl ModelGen for Model {
    fn get_v_segments(&self) -> Vec<Gene> {
        self.seg_vs.clone()
    }
    fn get_j_segments(&self) -> Vec<Gene> {
        self.seg_js.clone()
    }
}
