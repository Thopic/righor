use crate::sequence::{AlignmentParameters, AminoAcid, Dna};
use crate::shared::model::{sanitize_j, sanitize_v};
use crate::shared::parser::{parse_file, parse_str, ParserMarginals, ParserParams};
use crate::shared::utils::{
    add_errors, calc_steady_state_dist, sorted_and_complete, sorted_and_complete_0start,
    DiscreteDistribution, Gene, InferenceParameters, MarkovDNA,
};
use crate::vdj::inference::ResultInference;
use crate::vdj::sequence::{align_all_dgenes, align_all_jgenes, align_all_vgenes};
use crate::vdj::{Features, Sequence, StaticEvent};
use anyhow::{anyhow, Result};
use ndarray::{s, Array1, Array2, Array3, Axis};
use rand::Rng;
use std::path::Path;
use std::{cmp, fs::File};

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

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
    pub p_d_given_vj: Array3<f64>,
    pub p_j_given_v: Array2<f64>,
    pub p_v: Array1<f64>,
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
    pub p_dj: Array2<f64>,
    pub p_vdj: Array3<f64>,
    pub first_nt_bias_ins_vd: Array1<f64>,
    pub first_nt_bias_ins_dj: Array1<f64>,
    pub thymic_q: f64,
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

            model.p_v = Array1::<f64>::zeros(vdim);
            model.p_j_given_v = Array2::<f64>::zeros((jdim, vdim));
            model.p_d_given_vj = Array3::<f64>::zeros((ddim, vdim, jdim));
            for vv in 0..vdim {
                for jj in 0..jdim {
                    for dd in 0..ddim {
                        model.p_d_given_vj[[dd, vv, jj]] = pd[[vv, jj, dd]];
                    }
                    model.p_j_given_v[[jj, vv]] = pj[[vv, jj]];
                }
                model.p_v[[vv]] = pv[[vv]];
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

            model.p_v = Array1::<f64>::zeros(vdim);
            model.p_j_given_v = Array2::<f64>::zeros((jdim, vdim));
            model.p_d_given_vj = Array3::<f64>::zeros((ddim, vdim, jdim));

            for vv in 0..vdim {
                for jj in 0..jdim {
                    for dd in 0..ddim {
                        model.p_d_given_vj[[dd, vv, jj]] = pd[[jj, dd]];
                    }
                    model.p_j_given_v[[jj, vv]] = pj[[jj]];
                }
                model.p_v[[vv]] = pv[[vv]];
            }
        } else {
            return Err::<Model, anyhow::Error>(anyhow!("Wrong format for the VDJ probabilities"));
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
        model.p_del_d5_del_d3 = Array3::<f64>::zeros((d5dim, d3dim, ddim));
        for dd in 0..ddim {
            for d5 in 0..d5dim {
                for d3 in 0..d3dim {
                    model.p_del_d5_del_d3[[d5, d3, dd]] = pdeld3[[dd, d5, d3]] * pdeld5[[dd, d5]]
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

        model.error_rate = pp.error_rate;
        model.thymic_q = 9.41; // TODO: deal with this

        model.initialize()?;

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

        self.gen.markov_vd = MarkovDNA::new(self.markov_coefficients_vd.t().to_owned())?;
        self.gen.markov_dj = MarkovDNA::new(self.markov_coefficients_dj.t().to_owned())?;
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
        // define the non-critical parameters
        self.set_p_vdj(
            &self.p_d_given_vj.clone(),
            &self.p_j_given_v.clone(),
            &self.p_v.clone(),
        );
        self.initialize_generative_model()?;
        // load the data and normalize
        let mut feature = Features::new(self)?;
        feature.normalize()?;

        self.load_features(&feature)?;
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

    pub fn infer(
        &self,
        sequence: &Sequence,
        inference_params: &InferenceParameters,
    ) -> Result<ResultInference> {
        let mut ip = inference_params.clone();
        ip.nb_best_events = 0;
        ip.evaluate = true;
        let mut feature = Features::new(self)?;
        feature.infer(sequence, &ip)
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

    pub fn load_features(&mut self, feature: &Features) -> Result<()> {
        self.p_v = feature.v.log_probas.mapv(|x| x.exp2());
        self.p_del_v_given_v = feature.delv.log_probas.mapv(|x| x.exp2());
        self.set_p_vdj(
            &feature.d.log_probas.mapv(|x| x.exp2()),
            &feature.j.log_probas.mapv(|x| x.exp2()),
            &feature.v.log_probas.mapv(|x| x.exp2()),
        );
        self.p_del_j_given_j = feature.delj.log_probas.mapv(|x| x.exp2());
        self.p_del_d5_del_d3 = feature.deld.log_probas.mapv(|x| x.exp2());
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

    pub fn set_p_vdj(
        &mut self,
        p_d_given_vj: &Array3<f64>,
        p_j_given_v: &Array2<f64>,
        p_v: &Array1<f64>,
    ) {
        // P(V,D,J) = P(D | V, J) * P(V, J) = P(D|V,J) * P(J|V)*P(V)
        let dim = p_d_given_vj.dim();
        self.p_vdj = Array3::zeros((dim.1, dim.0, dim.2));
        self.p_dj = Array2::zeros((dim.0, dim.2));
        for vv in 0..p_v.dim() {
            for jj in 0..p_j_given_v.dim().0 {
                for dd in 0..p_d_given_vj.dim().0 {
                    self.p_vdj[[vv, dd, jj]] =
                        p_d_given_vj[[dd, vv, jj]] * p_j_given_v[[jj, vv]] * p_v[vv];
                    self.p_dj[[dd, jj]] += self.p_vdj[[vv, dd, jj]];
                }
            }
        }
        self.p_d_given_vj = p_d_given_vj.clone();
        self.p_j_given_v = p_j_given_v.clone();
        self.p_v = p_v.clone();
    }
}
