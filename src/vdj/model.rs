use crate::sequence::{AlignmentParameters, AminoAcid, Dna};
use crate::shared::model::{sanitize_j, sanitize_v};
use crate::shared::parser::{parse_file, parse_str, ParserMarginals, ParserParams};
use crate::shared::utils::{
    add_errors, calc_steady_state_dist, max_of_array, sorted_and_complete,
    sorted_and_complete_0start, DiscreteDistribution, Gene, InferenceParameters, MarkovDNA,
};
use crate::vdj::sequence::{align_all_dgenes, align_all_jgenes, align_all_vgenes};
use crate::vdj::{Features, Sequence, StaticEvent};
use anyhow::{anyhow, Result};
use ndarray::{s, Array1, Array2, Array3, Axis};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use rand::Rng;
use std::path::Path;
use std::{cmp, fs::File};

#[derive(Default, Clone, Debug)]
pub struct Generative {
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

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
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
    pub p_v: Array1<f64>,
    pub p_dj: Array2<f64>,
    pub p_ins_vd: Array1<f64>,
    pub p_ins_dj: Array1<f64>,
    pub p_del_v_given_v: Array2<f64>,
    pub p_del_j_given_j: Array2<f64>,
    pub p_del_d3_del_d5: Array3<f64>, // P(del_d3, del_d5 | D)
    pub gen: Generative,
    pub markov_coefficients_vd: Array2<f64>,
    pub markov_coefficients_dj: Array2<f64>,
    pub first_nt_bias_ins_vd: Array1<f64>,
    pub first_nt_bias_ins_dj: Array1<f64>,
    pub range_del_v: (i64, i64),
    pub range_del_j: (i64, i64),
    pub range_del_d3: (i64, i64),
    pub range_del_d5: (i64, i64),
    pub error_rate: f64,
    pub thymic_q: f64,

    // pre-computed values to improve speed
    pub max_log_likelihood_post_v: f64,
    pub max_log_likelihood_post_delv: f64,
    pub arr_max_log_likelihood_post_dj: Array2<f64>, // depend on d.index and the max number of insert
    pub arr_max_log_likelihood_post_delj: Array2<f64>, // depend on d.index and the number of inserts
    pub arr_max_log_likelihood_post_deld: Array2<f64>, // depend on the number of inserts on both sides of d
}

impl Model {
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

        model.initialize()?;
        model.error_rate = pp.error_rate;
        model.thymic_q = 9.41; // TODO: deal with this
        Ok(model)
    }
}

impl Model {
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

    pub fn generate_cdr3_no_error<R: Rng>(
        &mut self,
        functional: bool,
        rng: &mut R,
    ) -> (Dna, Option<AminoAcid>, StaticEvent) {
        // loop until we find a valid sequence (if generating functional alone)
        loop {
            let mut event = StaticEvent {
                ..Default::default()
            };

            event.v_index = self.gen.d_v.generate(rng);
            let dj_index: usize = self.gen.d_dj.generate(rng);
            event.d_index = dj_index / self.p_dj.dim().1;
            event.j_index = dj_index % self.p_dj.dim().1;

            let seq_v_cdr3: &Dna = &self.seg_vs_sanitized[event.v_index];
            let seq_j_cdr3: &Dna = &self.seg_js_sanitized[event.j_index];

            let seq_d: &Dna = self.seg_ds[event.d_index].seq_with_pal.as_ref().unwrap();
            let seq_v: &Dna = self.seg_vs[event.v_index].seq_with_pal.as_ref().unwrap();

            event.delv = self.gen.d_del_v_given_v[event.v_index].generate(rng);
            let del_d: usize = self.gen.d_del_d3_del_d5[event.d_index].generate(rng);
            event.deld5 = del_d / self.p_del_d3_del_d5.dim().0;
            event.deld3 = del_d % self.p_del_d3_del_d5.dim().0;
            event.delj = self.gen.d_del_j_given_j[event.j_index].generate(rng);

            let ins_vd: usize = self.gen.d_ins_vd.generate(rng);
            let ins_dj: usize = self.gen.d_ins_dj.generate(rng);

            let out_of_frame =
                (seq_v_cdr3.len() - event.delv + seq_d.len() - event.deld5 - event.deld3
                    + seq_j_cdr3.len()
                    - event.delj
                    + ins_vd
                    + ins_dj)
                    % 3
                    != 0;
            if functional && out_of_frame {
                continue;
            }

            let ins_seq_vd: Dna = self.gen.markov_vd.generate(ins_vd, rng);
            let mut ins_seq_dj: Dna = self.gen.markov_dj.generate(ins_dj, rng);
            ins_seq_dj.reverse(); // reverse for integration

            event.insdj = ins_seq_dj.clone();
            event.insvd = ins_seq_vd.clone();
            event.v_start_gene = 0;
            event.d_start_seq = seq_v.len() - event.delv - event.deld5 + ins_vd;
            event.j_start_seq = seq_v.len() - event.delv + ins_vd + ins_dj + seq_d.len()
                - event.deld5
                - event.deld3
                - event.delj;

            // create the complete CDR3 sequence
            let seq = event.to_cdr3(self);

            // translate
            let seq_aa: Option<AminoAcid> = seq.translate().ok();

            match seq_aa {
                Some(saa) => {
                    // check for stop codon
                    if functional && saa.seq.contains(&b'*') {
                        continue;
                    }

                    // check for conserved extremities (cysteine)
                    if functional && (saa.seq[0] != b'C') {
                        continue;
                    }
                    return (seq, Some(saa), event);
                }
                None => {
                    if functional {
                        continue;
                    }
                    return (seq, None, event);
                }
            }
        }
    }

    /// Return (cdr3_nt, cdr3_aa, full_sequence, event, vname, jname)
    pub fn generate<R: Rng>(
        &mut self,
        functional: bool,
        rng: &mut R,
    ) -> (Dna, Option<AminoAcid>, Dna, StaticEvent, String, String) {
        let (cdr3_nt, _, event) = self.generate_cdr3_no_error(functional, rng);
        let (mut full_seq, vname, jname, start_cdr3) =
            self.recreate_full_sequence(&cdr3_nt, event.v_index, event.j_index);
        // add potential sequencing error
        add_errors(&mut full_seq, self.error_rate, rng);
        let cdr3_err = full_seq.extract_subsequence(start_cdr3, start_cdr3 + cdr3_nt.len());
        let seq_aa: Option<AminoAcid> = cdr3_err.translate().ok();
        (cdr3_err, seq_aa, full_seq, event, vname, jname)
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl Model {
    pub fn uniform(&self) -> Result<Model> {
        let mut m = Model {
            seg_vs: self.seg_vs.clone(),
            seg_js: self.seg_js.clone(),
            seg_ds: self.seg_ds.clone(),
            range_del_d3: self.range_del_d3,
            range_del_v: self.range_del_v,
            range_del_j: self.range_del_j,
            range_del_d5: self.range_del_d5,
            p_v: Array1::<f64>::ones(self.p_v.dim()),
            p_dj: Array2::<f64>::ones(self.p_dj.dim()),
            p_ins_vd: Array1::<f64>::ones(self.p_ins_vd.dim()),
            p_ins_dj: Array1::<f64>::ones(self.p_ins_dj.dim()),
            p_del_v_given_v: Array2::<f64>::ones(self.p_del_v_given_v.dim()),
            p_del_j_given_j: Array2::<f64>::ones(self.p_del_j_given_j.dim()),
            p_del_d3_del_d5: Array3::<f64>::ones(self.p_del_d3_del_d5.dim()),
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
        self.initialize_generative_model()?;
        self.precompute_maximum_log_likelihoods();
        Ok(())
    }

    pub fn infer_features(
        &self,
        sequence: &Sequence,
        inference_params: &InferenceParameters,
    ) -> Result<Features> {
        let mut feature = Features::new(self)?;
        let _ = feature.infer(sequence, inference_params);
        Ok(feature)
    }
    // pub fn most_likely_recombinations(
    //     &self,
    //     sequence: &Sequence,
    //     nb_scenarios: usize,
    //     inference_params: &InferenceParameters,
    // ) -> Result<Vec<(f64, StaticEvent)>> {
    //     let mut ip = inference_params.clone();
    //     ip.nb_best_events = nb_scenarios;
    //     ip.evaluate = true;
    //     let mut feature = Features::new(self)?;
    //     let _, res) = feature.infer(sequence, self, &ip);
    //     Ok(res)
    // }
    pub fn pgen(&self, sequence: &Sequence, inference_params: &InferenceParameters) -> Result<f64> {
        let mut ip = inference_params.clone();
        ip.nb_best_events = 0;
        ip.evaluate = true;
        let mut feature = Features::new(self)?;
        let pg = feature.infer(sequence, &ip);
        pg
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

        // roughly estimate bounds for the position of d
        // TODO: not great, improve on that
        let left_bound = seq
            .v_genes
            .iter()
            .map(|v| {
                if v.end_seq > self.p_del_v_given_v.dim().0 + self.p_del_d3_del_d5.dim().1 {
                    v.end_seq - (self.p_del_v_given_v.dim().0 + self.p_del_d3_del_d5.dim().1)
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
                    j.start_seq + (self.p_del_j_given_j.dim().0 + self.p_del_d3_del_d5.dim().0),
                    dna_seq.len(),
                )
            })
            .max()
            .ok_or(anyhow!("Error in the definition of the D gene bounds"))?;

        // println!("pdeld3deld5 {:?}", self.p_del_d3_del_d5.dim());
        // println!("{} {}", left_bound, right_bound);
        // initialize all the d genes positions
        seq.d_genes = align_all_dgenes(&dna_seq, self, left_bound, right_bound, align_params);
        Ok(seq)
    }

    /// Re-create the full sequence of the variable region (with complete V/J gene, not just the CDR3)
    /// Return full_seq, v name, j name, cdr3_start
    pub fn recreate_full_sequence(
        &self,
        dna: &Dna,
        v_index: usize,
        j_index: usize,
    ) -> (Dna, String, String, usize) {
        let mut seq: Dna = Dna::new();
        let vgene = self.seg_vs[v_index].clone();
        let jgene = self.seg_js[j_index].clone();
        let vgene_sans_cdr3 = vgene.seq.extract_subsequence(0, vgene.cdr3_pos.unwrap());
        seq.extend(&vgene_sans_cdr3);
        seq.extend(dna);
        seq.extend(
            &jgene
                .seq
                .extract_subsequence(jgene.cdr3_pos.unwrap() + 3, jgene.seq.len()),
        );
        (seq, vgene.name, jgene.name, vgene_sans_cdr3.len())
    }

    pub fn update(&mut self, feature: &Features) -> Result<()> {
        self.p_v = feature.v.log_probas.mapv(|x| x.exp2());
        self.p_del_v_given_v = feature.delv.log_probas.mapv(|x| x.exp2());
        self.p_dj = feature.dj.log_probas.mapv(|x| x.exp2());
        self.p_del_j_given_j = feature.delj.log_probas.mapv(|x| x.exp2());
        self.p_del_d3_del_d5 = feature.deld.log_probas.mapv(|x| x.exp2());
        (
            self.p_ins_vd,
            self.first_nt_bias_ins_vd,
            self.markov_coefficients_vd,
        ) = feature.insvd.get_parameters();
        (
            self.p_ins_dj,
            self.first_nt_bias_ins_dj,
            self.markov_coefficients_dj,
        ) = feature.insdj.get_parameters();
        self.error_rate = feature.error.error_rate;
        self.initialize()?;
        Ok(())
    }

    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[staticmethod]
    #[pyo3(name = "load_model")]
    pub fn py_load_model(
        path_params: &str,
        path_marginals: &str,
        path_anchor_vgene: &str,
        path_anchor_jgene: &str,
    ) -> Result<Model> {
        Model::load_from_files(
            Path::new(path_params),
            Path::new(path_marginals),
            Path::new(path_anchor_vgene),
            Path::new(path_anchor_jgene),
        )
    }

    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_v(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_v.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_v(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_v = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_dj(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_dj.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_dj(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_dj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_ins_vd(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_ins_vd.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_ins_vd(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_ins_vd = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_ins_dj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.p_ins_dj.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_ins_dj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.p_ins_dj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_del_v_given_v(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_del_v_given_v.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_del_v_given_v(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_del_v_given_v = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_del_j_given_j(&self, py: Python) -> Py<PyArray2<f64>> {
        self.p_del_j_given_j.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_del_j_given_j(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.p_del_j_given_j = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_p_del_d3_del_d5(&self, py: Python) -> Py<PyArray3<f64>> {
        self.p_del_d3_del_d5.to_owned().into_pyarray(py).to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_p_del_d3_del_d5(&mut self, py: Python, value: Py<PyArray3<f64>>) -> PyResult<()> {
        self.p_del_d3_del_d5 = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_markov_coefficients_vd(&self, py: Python) -> Py<PyArray2<f64>> {
        self.markov_coefficients_vd
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_markov_coefficients_vd(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.markov_coefficients_vd = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_markov_coefficients_dj(&self, py: Python) -> Py<PyArray2<f64>> {
        self.markov_coefficients_dj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_markov_coefficients_dj(&mut self, py: Python, value: Py<PyArray2<f64>>) -> PyResult<()> {
        self.markov_coefficients_dj = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_first_nt_bias_ins_vd(&self, py: Python) -> Py<PyArray1<f64>> {
        self.first_nt_bias_ins_vd
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_first_nt_bias_ins_vd(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.first_nt_bias_ins_vd = value.as_ref(py).to_owned_array();
        Ok(())
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[getter]
    fn get_first_nt_bias_ins_dj(&self, py: Python) -> Py<PyArray1<f64>> {
        self.first_nt_bias_ins_dj
            .to_owned()
            .into_pyarray(py)
            .to_owned()
    }
    #[cfg(all(feature = "py_binds", feature = "pyo3"))]
    #[setter]
    fn set_first_nt_bias_ins_dj(&mut self, py: Python, value: Py<PyArray1<f64>>) -> PyResult<()> {
        self.first_nt_bias_ins_dj = value.as_ref(py).to_owned_array();
        Ok(())
    }
}

/// These functions compute max_log_likelihood values that are used
/// to get rid of dead branches during the inference.
impl Model {
    pub fn max_log_likelihood_post_dj(&self, d_idx: usize, nb_ins: i64) -> f64 {
        if nb_ins >= self.arr_max_log_likelihood_post_dj.dim().1 as i64 {
            return f64::NEG_INFINITY;
        } else if nb_ins < 0 {
            return self.arr_max_log_likelihood_post_dj[[d_idx, 0]];
        }
        self.arr_max_log_likelihood_post_dj[[d_idx, nb_ins as usize]]
    }

    pub fn max_log_likelihood_post_delj(&self, d_idx: usize, nb_ins: usize) -> f64 {
        if nb_ins >= self.arr_max_log_likelihood_post_delj.dim().1 {
            return f64::NEG_INFINITY;
        }
        self.arr_max_log_likelihood_post_delj[[d_idx, nb_ins]]
    }

    pub fn max_log_likelihood_post_deld(&self, nb_ins_vd: usize, nb_ins_dj: usize) -> f64 {
        if nb_ins_vd >= self.arr_max_log_likelihood_post_deld.dim().0
            || nb_ins_dj >= self.arr_max_log_likelihood_post_deld.dim().1
        {
            return f64::NEG_INFINITY;
        }
        self.arr_max_log_likelihood_post_deld[[nb_ins_vd, nb_ins_dj]]
    }

    pub fn precompute_maximum_log_likelihoods(&mut self) {
        let max_delv = max_of_array(&self.p_del_v_given_v).log2();
        let max_dj = max_of_array(&self.p_dj).log2();
        let max_delj = max_of_array(&self.p_del_j_given_j).log2();

        let max_lins = max_of_array(&self.p_ins_vd).log2() + max_of_array(&self.p_ins_dj).log2();
        let max_deld = max_of_array(&self.p_del_d3_del_d5).log2();

        let max_transition_vd = max_of_array(&self.markov_coefficients_vd).log2();
        let max_transition_dj = max_of_array(&self.markov_coefficients_dj).log2();

        let maxlend = self
            .seg_ds
            .iter()
            .map(|x| x.seq_with_pal.clone().unwrap().len())
            .max()
            .unwrap_or(0);

        self.max_log_likelihood_post_v = max_delv + max_delj + max_dj + max_lins + max_deld;
        self.max_log_likelihood_post_delv = max_dj + max_delj + max_lins + max_deld;

        self.arr_max_log_likelihood_post_deld =
            Array2::zeros((self.p_ins_vd.dim(), self.p_ins_dj.dim()));
        self.arr_max_log_likelihood_post_dj = Array2::zeros((
            self.p_dj.dim().0,
            self.p_ins_dj.dim() + self.p_ins_vd.dim() + maxlend,
        ));
        self.arr_max_log_likelihood_post_delj = Array2::zeros((
            self.p_dj.dim().0,
            self.p_ins_dj.dim() + self.p_ins_vd.dim() + maxlend,
        ));

        let mut max_ins = Array1::zeros(self.p_ins_vd.dim() + self.p_ins_dj.dim() - 1);
        max_ins += f64::NEG_INFINITY;

        for nb_ins_vd in 0..self.p_ins_vd.dim() {
            for nb_ins_dj in 0..self.p_ins_dj.dim() {
                let t = max_transition_vd * (nb_ins_vd as f64)
                    + self.p_ins_vd[nb_ins_vd].log2()
                    + max_transition_dj * (nb_ins_dj as f64)
                    + self.p_ins_dj[nb_ins_dj].log2();
                //                println!("{:?} {} {}", t, nb_ins_vd, nb_ins_dj);
                self.arr_max_log_likelihood_post_deld[[nb_ins_vd, nb_ins_dj]] = t;
                for a in 0..(nb_ins_vd + nb_ins_dj) + 1 {
                    if max_ins[a] < t {
                        max_ins[a] = t;
                    }
                }
            }
        }

        for d_idx in 0..self.p_dj.dim().0 {
            let lend = self.seg_ds[d_idx].seq_with_pal.clone().unwrap().len();
            for nb_ins in 0..self.p_ins_dj.dim() + self.p_ins_vd.dim() - 1 {
                self.arr_max_log_likelihood_post_dj[[d_idx, nb_ins + lend]] =
                    max_ins[nb_ins] + max_delj;
                self.arr_max_log_likelihood_post_delj[[d_idx, nb_ins + lend]] = max_ins[nb_ins];
            }
            for leftover in 0..maxlend - lend + 1 {
                self.arr_max_log_likelihood_post_delj[[
                    d_idx,
                    self.p_ins_dj.dim() + self.p_ins_vd.dim() + maxlend - 1 - leftover,
                ]] = f64::NEG_INFINITY;
                self.arr_max_log_likelihood_post_dj[[
                    d_idx,
                    self.p_ins_dj.dim() + self.p_ins_vd.dim() + maxlend - 1 - leftover,
                ]] = f64::NEG_INFINITY;
            }
        }
    }
}
