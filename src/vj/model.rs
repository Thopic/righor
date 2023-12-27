use crate::sequence::{AlignmentParameters, AminoAcid, Dna};
use crate::shared::model::{sanitize_j, sanitize_v};
use crate::shared::parser::{ParserMarginals, ParserParams};
use crate::shared::utils::{
    add_errors, sorted_and_complete, sorted_and_complete_0start, DiscreteDistribution, Gene,
    MarkovDNA,
};
use crate::shared::InferenceParameters;
use crate::vj::sequence::{align_all_jgenes, align_all_vgenes};
use crate::vj::{Features, Sequence, StaticEvent};
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Axis};
#[cfg(all(feature = "py_binds", feature = "py_o3"))]
use numpy::{IntoPyArray, PyArray1, PyArray2};
#[cfg(all(feature = "py_binds", feature = "py_o3"))]
use pyo3::prelude::*;
use rand::Rng;
use std::path::Path;

#[cfg_attr(features = "py_binds", pyclass)]
#[derive(Default, Clone, Debug)]
struct Generative {
    // Contains the distribution needed to generate the model
    d_v: DiscreteDistribution,
    d_j_given_v: Vec<DiscreteDistribution>,
    d_ins_vj: DiscreteDistribution,
    d_del_v_given_v: Vec<DiscreteDistribution>,
    d_del_j_given_j: Vec<DiscreteDistribution>,
    markov_vj: MarkovDNA,
}

#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pyclass)]
#[derive(Default, Clone, Debug)]
pub struct Model {
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
    gen: Generative,
    pub markov_coefficients_vj: Array2<f64>,
    pub first_nt_bias_ins_vj: Array1<f64>,
    pub range_del_v: (i64, i64),
    pub range_del_j: (i64, i64),
    pub error_rate: f64,
    pub thymic_q: f64,
}

impl Model {
    pub fn load_model(
        path_params: &Path,
        path_marginals: &Path,
        path_anchor_vgene: &Path,
        path_anchor_jgene: &Path,
    ) -> Result<Model> {
        let mut model: Model = Default::default();
        let pm: ParserMarginals = ParserMarginals::parse(path_marginals)?;
        let mut pp: ParserParams = ParserParams::parse(path_params)?;
        pp.add_anchors_gene(path_anchor_vgene, "v_choice")?;
        pp.add_anchors_gene(path_anchor_jgene, "j_choice")?;

        model.seg_vs = pp
            .params
            .get("v_choice")
            .ok_or(anyhow!("Error with unwrapping the Params data"))?
            .clone()
            .to_genes()?;
        model.seg_js = pp
            .params
            .get("j_choice")
            .ok_or(anyhow!("Error with unwrapping the Params data"))?
            .clone()
            .to_genes()?;

        let arrdelv = pp
            .params
            .get("v_3_del")
            .ok_or(anyhow!("Invalid v_del"))?
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

        model.sanitize_genes()?;

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

        // generative model
        model.initialize_generative_model()?;

        model.error_rate = pp.error_rate;
        model.thymic_q = 9.41; // TODO: deal with this
        Ok(model)
    }

    fn sanitize_genes(&mut self) -> Result<()> {
        // Trim the V/J nucleotides sequences at the CDR3 region (include F/W/C residues)
        // and append the maximum number of reverse palindromic insertions appended.
        // Add the palindromic insertions

        for g in self.seg_vs.iter_mut() {
            g.create_palindromic_ends(0, (-self.range_del_v.0) as usize);
        }
        for g in self.seg_js.iter_mut() {
            g.create_palindromic_ends((-self.range_del_j.0) as usize, 0);
        }

        self.seg_vs_sanitized = sanitize_v(self.seg_vs.clone())?;
        self.seg_js_sanitized = sanitize_j(self.seg_js.clone(), (-self.range_del_j.0) as usize)?;
        Ok(())
    }

    fn initialize_generative_model(&mut self) -> Result<()> {
        self.gen.d_v = DiscreteDistribution::new(self.p_v.to_vec())?;
        self.gen.d_ins_vj = DiscreteDistribution::new(self.p_ins_vj.to_vec())?;

        self.gen.d_j_given_v = Vec::new();
        for row in self.p_j_given_v.axis_iter(Axis(1)) {
            self.gen
                .d_j_given_v
                .push(DiscreteDistribution::new(row.to_vec())?);
        }

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

        self.gen.markov_vj = MarkovDNA::new(self.markov_coefficients_vj.t().to_owned(), None)?;
        Ok(())
    }

    pub fn generate<R: Rng>(
        &mut self,
        functional: bool,
        rng: &mut R,
    ) -> (Dna, Option<AminoAcid>, usize, usize) {
        // loop until we find a valid sequence (if generating functional alone)
        loop {
            let v_index: usize = self.gen.d_v.generate(rng);
            let j_index: usize = self.gen.d_j_given_v[v_index].generate(rng);

            let seq_v: &Dna = &self.seg_vs_sanitized[v_index];
            let seq_j: &Dna = &self.seg_js_sanitized[j_index];

            let del_v: usize = self.gen.d_del_v_given_v[v_index].generate(rng);
            let del_j: usize = self.gen.d_del_j_given_j[j_index].generate(rng);

            let ins_vj: usize = self.gen.d_ins_vj.generate(rng);

            let out_of_frame = (seq_v.len() - del_v + seq_j.len() - del_j + ins_vj) % 3 != 0;
            if functional & out_of_frame {
                continue;
            }

            let ins_seq_vj: Dna = self.gen.markov_vj.generate(ins_vj, rng);

            // create the complete sequence
            let mut seq: Dna = Dna::new();
            seq.extend(&seq_v.extract_subsequence(0, seq_v.len() - del_v));
            seq.extend(&ins_seq_vj);
            seq.extend(&seq_j.extract_subsequence(del_j, seq_j.len()));

            // add potential sequencing error
            add_errors(&mut seq, self.error_rate, rng);

            // translate
            let seq_aa: Option<AminoAcid> = seq.translate().ok();

            match seq_aa {
                Some(saa) => {
                    // check for stop codon
                    if functional & saa.seq.contains(&b'*') {
                        continue;
                    }

                    // check for conserved extremities (cysteine)
                    if functional & (saa.seq[0] != b'C') {
                        continue;
                    }
                    return (seq, Some(saa), v_index, j_index);
                }
                None => {
                    if functional {
                        continue;
                    }
                    return (seq, None, v_index, j_index);
                }
            }
        }
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pymethods)]
impl Model {
    pub fn infer_features(
        &self,
        sequence: &Sequence,
        inference_params: &InferenceParameters,
    ) -> Result<Features> {
        let mut feature = Features::new(self, inference_params)?;
        let _ = feature.infer(sequence, inference_params, 0);
        feature = feature.cleanup()?;
        Ok(feature)
    }
    pub fn most_likely_recombinations(
        &self,
        sequence: &Sequence,
        nb_scenarios: usize,
        inference_params: &InferenceParameters,
    ) -> Result<Vec<(f64, StaticEvent)>> {
        let mut feature = Features::new(self, inference_params)?;
        let (_, res) = feature.infer(sequence, inference_params, nb_scenarios);
        Ok(res)
    }
    pub fn pgen(&self, sequence: &Sequence, inference_params: &InferenceParameters) -> Result<f64> {
        let mut feature = Features::new(self, inference_params)?;
        let (pg, _) = feature.infer(sequence, inference_params, 0);
        Ok(pg)
    }

    pub fn align_sequence(
        &self,
        dna_seq: Dna,
        align_params: &AlignmentParameters,
    ) -> Result<Sequence> {
        let seq = Sequence {
            sequence: dna_seq.clone(),
            v_genes: align_all_vgenes(&dna_seq, self, align_params),
            j_genes: align_all_jgenes(&dna_seq, self, align_params),
            d_genes: Vec::new(),
        };
        Ok(seq)
    }

    pub fn recreate_full_sequence(
        &self,
        dna: &Dna,
        v_index: usize,
        j_index: usize,
    ) -> (Dna, String, String) {
        // Re-create the full sequence of the variable region (with complete V/J gene, not just the CDR3)
        let mut seq: Dna = Dna::new();
        let vgene = self.seg_vs[v_index].clone();
        let jgene = self.seg_js[j_index].clone();
        seq.extend(&vgene.seq.extract_subsequence(0, vgene.cdr3_pos.unwrap()));
        seq.extend(dna);
        seq.extend(
            &jgene
                .seq
                .extract_subsequence(jgene.cdr3_pos.unwrap() + 3, jgene.seq.len()),
        );
        (seq, vgene.name, jgene.name)
    }

    pub fn update(&mut self, feature: &Features) {
        self.p_v = feature.v.probas.clone();
        self.p_del_v_given_v = feature.delv.probas.clone();
        self.p_j_given_v = feature.j.probas.clone();
        self.p_del_j_given_j = feature.delj.probas.clone();
        self.p_ins_vj = feature.nb_insvj.probas.clone();
        (self.first_nt_bias_ins_vj, self.markov_coefficients_vj) = feature.insvj.get_parameters();
        self.error_rate = feature.error.error_rate;
    }

    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[staticmethod]
    #[pyo3(name = "load_model")]
    pub fn py_load_model(
        path_params: &str,
        path_marginals: &str,
        path_anchor_vgene: &str,
        path_anchor_jgene: &str,
    ) -> Result<Model> {
        Model::load_model(
            Path::new(path_params),
            Path::new(path_marginals),
            Path::new(path_anchor_vgene),
            Path::new(path_anchor_jgene),
        )
    }

    // getter & setter for the numpy/ndarray arrays, no easy way to make them automatically
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[getter]
    fn get_p_v(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_v.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[setter]
    fn set_p_v(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_v = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[getter]
    fn get_p_j_given_v(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_j_given_v.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[setter]
    fn set_p_j_given_v(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_j_given_v = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[getter]
    fn get_p_ins_vj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_ins_vj.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[setter]
    fn set_p_ins_vj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_ins_vj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[getter]
    fn get_p_del_v_given_v(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_del_v_given_v.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[setter]
    fn set_p_del_v_given_v(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_del_v_given_v = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[getter]
    fn get_p_del_j_given_j(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_del_j_given_j.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[setter]
    fn set_p_del_j_given_j(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_del_j_given_j = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[getter]
    fn get_markov_coefficients_vj(&self, py: Python) -> Py<PyArray2<f64>> {
        self.markov_coefficients_vj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[setter]
    fn set_markov_coefficients_vj(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.markov_coefficients_vj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[getter]
    fn get_first_nt_bias_ins_vj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.first_nt_bias_ins_vj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "py_o3"))]
    #[setter]
    fn set_first_nt_bias_ins_vj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.first_nt_bias_ins_vj = value.as_ref(py).to_owned_array();
        Ok(())
    }
}
