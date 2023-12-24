use crate::parser::{ParserMarginals, ParserParams};
use crate::utils::{
    add_errors, calc_steady_state_dist, sorted_and_complete, sorted_and_complete_0start,
    DiscreteDistribution, Gene, MarkovDNA,
};
use crate::utils_sequences::{AminoAcid, Dna};
use anyhow::{anyhow, Result};
use duplicate::duplicate_item;
use ndarray::{s, Array1, Array2, Array3, Axis};
use pyo3::prelude::*;
use rand::Rng;
use std::path::Path;

#[derive(Default, Clone, Debug)]
struct GenerativeVDJ {
    // Contains the distribution needed to generate the model
    d_v: DiscreteDistribution,
    d_dj: DiscreteDistribution,
    d_ins_vd: DiscreteDistribution,
    d_ins_dj: DiscreteDistribution,
    d_del_v_given_v: Vec<DiscreteDistribution>,
    d_del_j_given_j: Vec<DiscreteDistribution>,
    d_del_d3_del_d5: Vec<DiscreteDistribution>,
    markov_vd: MarkovDNA,
    markov_dj: MarkovDNA,
}

#[pyclass]
#[derive(Default, Clone, Debug)]
pub struct ModelVDJ {
    // Sequence information
    #[pyo3(get, set)]
    pub seg_vs: Vec<Gene>,
    #[pyo3(get, set)]
    pub seg_js: Vec<Gene>,
    #[pyo3(get, set)]
    pub seg_ds: Vec<Gene>,

    // V/J sequences trimmed at the CDR3 region (include F/W/C residues) with
    // the maximum number of reverse palindromic insertions appended.
    #[pyo3(get, set)]
    seg_vs_sanitized: Vec<Dna>, // match genomic_data.cutV_genomic_CDR3_seqs
    #[pyo3(get, set)]
    seg_js_sanitized: Vec<Dna>,

    // Probabilities of the different events
    pub p_v: Array1<f64>,
    pub p_dj: Array2<f64>,
    pub p_ins_vd: Array1<f64>,
    pub p_ins_dj: Array1<f64>,
    pub p_del_v_given_v: Array2<f64>,
    pub p_del_j_given_j: Array2<f64>,
    pub p_del_d3_del_d5: Array3<f64>,
    gen: GenerativeVDJ,
    pub markov_coefficients_vd: Array2<f64>,
    pub markov_coefficients_dj: Array2<f64>,
    pub first_nt_bias_ins_vd: Array1<f64>,
    pub first_nt_bias_ins_dj: Array1<f64>,
    #[pyo3(get, set)]
    pub max_del_v: usize,
    #[pyo3(get, set)]
    pub max_del_j: usize,
    #[pyo3(get, set)]
    pub max_del_d3: usize,
    #[pyo3(get, set)]
    pub max_del_d5: usize,
    #[pyo3(get, set)]
    pub error_rate: f64,
    #[pyo3(get, set)]
    thymic_q: f64,
}

#[derive(Default, Clone, Debug)]
struct GenerativeVJ {
    // Contains the distribution needed to generate the model
    d_v: DiscreteDistribution,
    d_j_given_v: Vec<DiscreteDistribution>,
    d_ins_vj: DiscreteDistribution,
    d_del_v_given_v: Vec<DiscreteDistribution>,
    d_del_j_given_j: Vec<DiscreteDistribution>,
    markov_vj: MarkovDNA,
}

#[pyclass]
#[derive(Default, Clone, Debug)]
pub struct ModelVJ {
    // Sequence information
    #[pyo3(get, set)]
    pub seg_vs: Vec<Gene>,
    #[pyo3(get, set)]
    pub seg_js: Vec<Gene>,

    // V/J nucleotides sequences trimmed at the CDR3 region (include F/W/C residues) with
    // the maximum number of reverse palindromic insertions appended.
    #[pyo3(get, set)]
    pub seg_vs_sanitized: Vec<Dna>, // match genomic_data.cutV_genomic_CDR3_seqs
    #[pyo3(get, set)]
    pub seg_js_sanitized: Vec<Dna>,

    // Probabilities of the different events
    pub p_v: Array1<f64>,
    pub p_j_given_v: Array2<f64>,
    pub p_ins_vj: Array1<f64>,
    pub p_del_v_given_v: Array2<f64>,
    pub p_del_j_given_j: Array2<f64>,
    gen: GenerativeVJ,
    pub markov_coefficients_vj: Array2<f64>,
    pub first_nt_bias_ins_vj: Array1<f64>,
    #[pyo3(get, set)]
    pub max_del_v: usize,
    #[pyo3(get, set)]
    pub max_del_j: usize,
    #[pyo3(get, set)]
    pub error_rate: f64,
    #[pyo3(get, set)]
    pub thymic_q: f64,
}

#[duplicate_item(model; [ModelVDJ]; [ModelVJ])]
impl model {
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
}

impl ModelVDJ {
    pub fn load_model(
        path_params: &Path,
        path_marginals: &Path,
        path_anchor_vgene: &Path,
        path_anchor_jgene: &Path,
    ) -> Result<ModelVDJ> {
        let mut model: ModelVDJ = Default::default();
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
        model.seg_ds = pp
            .params
            .get("d_gene")
            .ok_or(anyhow!("Error with unwrapping the Params data"))?
            .clone()
            .to_genes()?;

        let arrdelv = pp
            .params
            .get("v_3_del")
            .ok_or(anyhow!("Invalid v_3_del"))?
            .clone()
            .to_numbers()?;
        model.max_del_v = (-arrdelv.iter().min().ok_or(anyhow!("Empty v_3_del"))?)
            .try_into()
            .map_err(|_e| anyhow!("Invalid v_3_del"))?;
        let arrdelj = pp
            .params
            .get("j_5_del")
            .ok_or(anyhow!("Invalid j_5_del"))?
            .clone()
            .to_numbers()?;
        model.max_del_j = (-arrdelj.iter().min().ok_or(anyhow!("Empty j_5_del"))?)
            .try_into()
            .map_err(|_e| anyhow!("Invalid j_5_del"))?;
        let arrdeld3 = pp
            .params
            .get("d_3_del")
            .ok_or(anyhow!("Invalid d_3_del"))?
            .clone()
            .to_numbers()?;
        model.max_del_d3 = (-arrdeld3.iter().min().ok_or(anyhow!("Empty d_3_del"))?)
            .try_into()
            .map_err(|_e| anyhow!("Invalid d_3_del"))?;
        let arrdeld5 = pp
            .params
            .get("d_5_del")
            .ok_or(anyhow!("Invalid d_5_del"))?
            .clone()
            .to_numbers()?;
        model.max_del_d5 = (-arrdeld5.iter().min().ok_or(anyhow!("Empty d_5_del"))?)
            .try_into()
            .map_err(|_e| anyhow!("Invalid d_5_del"))?;

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
        model.p_v = pm
            .marginals
            .get("v_choice")
            .unwrap()
            .probabilities
            .clone()
            .into_dimensionality()
            .unwrap();
        // For the joint probability P(D, J), we just multiply
        // P(J) and P(D|J)
        // model.p_dj[d, j] = pj[j] * pd[j, d]
        let pd = pm.marginals.get("d_gene").unwrap().probabilities.clone();
        let pj = pm.marginals.get("j_choice").unwrap().probabilities.clone();
        let jdim = pd.dim()[0];
        let ddim = pd.dim()[1];
        model.p_dj = Array2::<f64>::zeros((ddim, jdim));
        for dd in 0..ddim {
            for jj in 0..jdim {
                model.p_dj[[dd, jj]] = pd[[jj, dd]] * pj[[jj]]
            }
        }

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
        // compute the joint probability P(delD3, delD5 | D)
        // P(delD3, delD5 | D) = P(delD3 | delD5, D) * P(delD5 | D)
        let pdeld3 = pm.marginals.get("d_3_del").unwrap().probabilities.clone(); // P(deld3| delD5, D)
        let pdeld5 = pm.marginals.get("d_5_del").unwrap().probabilities.clone(); // P(deld5| D)
        let ddim = pdeld3.dim()[0];
        let d5dim = pdeld3.dim()[1];
        let d3dim = pdeld3.dim()[2];
        model.p_del_d3_del_d5 = Array3::<f64>::zeros((d5dim, d3dim, ddim));
        for dd in 0..ddim {
            for d5 in 0..d5dim {
                for d3 in 0..d3dim {
                    model.p_del_d3_del_d5[[d5, d3, dd]] = pdeld3[[dd, d5, d3]] * pdeld5[[dd, d5]]
                }
            }
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

        // generative model
        model.initialize_generative_model()?;

        model.error_rate = pp.error_rate;
        model.thymic_q = 9.41; // TODO: deal with this
        Ok(model)
    }
}

impl ModelVDJ {
    fn sanitize_genes(&mut self) -> Result<()> {
        // Trim the V/J nucleotides sequences at the CDR3 region (include F/W/C residues)
        // and append the maximum number of reverse palindromic insertions appended.

        // Add the palindromic insertions
        for g in self.seg_vs.iter_mut() {
            g.create_palindromic_ends(0, self.max_del_v);
        }
        for g in self.seg_js.iter_mut() {
            g.create_palindromic_ends(self.max_del_j, 0);
        }
        for g in self.seg_ds.iter_mut() {
            g.create_palindromic_ends(self.max_del_d5, self.max_del_d3);
        }

        // cut the V/J at the CDR3 region
        self.seg_vs_sanitized = sanitize_v(self.seg_vs.clone())?;
        self.seg_js_sanitized = sanitize_j(self.seg_js.clone(), self.max_del_j)?;
        Ok(())
    }

    fn initialize_generative_model(&mut self) -> Result<()> {
        self.gen.d_v = DiscreteDistribution::new(self.p_v.to_vec())?;
        self.gen.d_dj = DiscreteDistribution::new(self.p_dj.view().iter().cloned().collect())?;
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

        self.gen.d_del_d3_del_d5 = Vec::new();
        for ii in 0..self.p_del_d3_del_d5.dim().2 {
            let d3d5: Vec<f64> = self
                .p_del_d3_del_d5
                .slice(s![.., .., ii])
                .iter()
                .cloned()
                .collect();
            self.gen
                .d_del_d3_del_d5
                .push(DiscreteDistribution::new(d3d5)?);
        }

        self.gen.markov_vd = MarkovDNA::new(self.markov_coefficients_vd.t().to_owned(), None)?;
        self.gen.markov_dj = MarkovDNA::new(self.markov_coefficients_dj.t().to_owned(), None)?;
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
            let dj_index: usize = self.gen.d_dj.generate(rng);
            let d_index: usize = dj_index / self.p_dj.dim().1;
            let j_index: usize = dj_index % self.p_dj.dim().1;

            let seq_v: &Dna = &self.seg_vs_sanitized[v_index];
            let seq_d: &Dna = self.seg_ds[d_index].seq_with_pal.as_ref().unwrap();
            let seq_j: &Dna = &self.seg_js_sanitized[j_index];

            let del_v: usize = self.gen.d_del_v_given_v[v_index].generate(rng);
            let del_d: usize = self.gen.d_del_d3_del_d5[d_index].generate(rng);
            let del_d5: usize = del_d / self.p_del_d3_del_d5.dim().0;
            let del_d3: usize = del_d % self.p_del_d3_del_d5.dim().0;
            let del_j: usize = self.gen.d_del_j_given_j[j_index].generate(rng);

            let ins_vd: usize = self.gen.d_ins_vd.generate(rng);
            let ins_dj: usize = self.gen.d_ins_dj.generate(rng);

            let out_of_frame = (seq_v.len() - del_v + seq_d.len() - del_d5 - del_d3 + seq_j.len()
                - del_j
                + ins_vd
                + ins_dj)
                % 3
                != 0;
            if functional & out_of_frame {
                continue;
            }

            let ins_seq_vd: Dna = self.gen.markov_vd.generate(ins_vd, rng);
            let mut ins_seq_dj: Dna = self.gen.markov_dj.generate(ins_dj, rng);
            ins_seq_dj.reverse(); // reverse for integration

            // create the complete sequence
            let mut seq: Dna = Dna::new();
            seq.extend(&seq_v.extract_subsequence(0, seq_v.len() - del_v));
            seq.extend(&ins_seq_vd);
            seq.extend(&seq_d.extract_subsequence(del_d5, seq_d.len() - del_d3));
            seq.extend(&ins_seq_dj);
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

impl ModelVJ {
    pub fn load_model(
        path_params: &Path,
        path_marginals: &Path,
        path_anchor_vgene: &Path,
        path_anchor_jgene: &Path,
    ) -> Result<ModelVJ> {
        let mut model: ModelVJ = Default::default();
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
            .ok_or(anyhow!("Invalid v_3_del"))?
            .clone()
            .to_numbers()?;
        model.max_del_v = (-arrdelv.iter().min().ok_or(anyhow!("Empty v_3_del"))?)
            .try_into()
            .map_err(|_e| anyhow!("Invalid v_3_del"))?;
        let arrdelj = pp
            .params
            .get("j_5_del")
            .ok_or(anyhow!("Invalid j_5_del"))?
            .clone()
            .to_numbers()?;
        model.max_del_j = (-arrdelj.iter().min().ok_or(anyhow!("Empty j_5_del"))?)
            .try_into()
            .map_err(|_e| anyhow!("Invalid j_5_del"))?;

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
            g.create_palindromic_ends(0, self.max_del_v);
        }
        for g in self.seg_js.iter_mut() {
            g.create_palindromic_ends(self.max_del_j, 0);
        }

        self.seg_vs_sanitized = sanitize_v(self.seg_vs.clone())?;
        self.seg_js_sanitized = sanitize_j(self.seg_js.clone(), self.max_del_j)?;
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

fn sanitize_v(genes: Vec<Gene>) -> Result<Vec<Dna>> {
    // Add palindromic inserted nucleotides to germline V sequences and cut all
    // sequences to only keep their CDR3 parts
    let mut cut_genes = Vec::<Dna>::new();
    for g in genes {
        // some V-genes are not complete. They don't appear in the model, but we
        // can't ignore them
        // TODO: I need to change the way this is done...
        if g.cdr3_pos.unwrap() >= g.seq.len() {
            cut_genes.push(Dna::new());
            continue;
        }

        let gene_seq: Dna = g
            .seq_with_pal
            .ok_or(anyhow!("Palindromic sequences not created"))?;

        let cut_gene: Dna = Dna {
            seq: gene_seq.seq[g.cdr3_pos.unwrap()..].to_vec(),
        };
        cut_genes.push(cut_gene);
    }
    Ok(cut_genes)
}

fn sanitize_j(genes: Vec<Gene>, max_del_j: usize) -> Result<Vec<Dna>> {
    // Add palindromic inserted nucleotides to germline J sequences and cut all
    // sequences to only keep their CDR3 parts
    let mut cut_genes = Vec::<Dna>::new();
    for g in genes {
        let gene_seq: Dna = g
            .seq_with_pal
            .ok_or(anyhow!("Palindromic sequences not created"))?;

        // for J, we want to also add the last CDR3 amino-acid (F/W)
        let cut_gene: Dna = Dna {
            seq: gene_seq.seq[..g.cdr3_pos.unwrap() + 3 + max_del_j].to_vec(),
        };
        cut_genes.push(cut_gene);
    }
    Ok(cut_genes)
}
