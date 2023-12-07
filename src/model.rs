use crate::parser::{ParserMarginals, ParserParams};
use crate::utils::{DiscreteDistribution, MarkovDNA, Gene, Dna, AminoAcid, add_errors};
use ndarray::{Array1, Array2, Array3, Axis, s};
use std::path::Path;
use rand::Rng;
use duplicate::duplicate_item;
use std::error::Error;



#[derive(Default, Clone, Debug)]
struct GenerativeVDJ{
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


#[derive(Default, Clone, Debug)]
pub struct ModelVDJ{
    // Sequence information
    seg_vs: Vec<Gene>,
    seg_js: Vec<Gene>,
    seg_ds: Vec<Gene>,

    // V/J nucleotides sequences trimmed at the CDR3 region (include F/W/C residues) with
    // the maximum number of reverse palindromic insertions appended.
    pub seg_vs_sanitized: Vec<Dna>, // match genomic_data.cutV_genomic_CDR3_seqs
    seg_js_sanitized: Vec<Dna>,
    seg_ds_sanitized: Vec<Dna>,

    // Probabilities of the different events
    pub p_v: Array1<f64>,
    pub p_dj: Array2<f64>,
    pub p_ins_vd: Array1<f64>,
    pub p_ins_dj: Array1<f64>,
    pub p_del_v_given_v: Array2<f64>,
    pub p_del_j_given_j: Array2<f64>,
    pub p_del_d3_del_d5: Array3<f64>,
    gen: GenerativeVDJ,
    markov_coefficients_vd: Array2<f64>,
    markov_coefficients_dj: Array2<f64>,
    first_nt_bias_ins_vd: Array2<f64>,
    first_nt_bias_ins_dj: Array2<f64>,
    max_del_v: usize,
    max_del_j: usize,
    max_del_d3: usize,
    max_del_d5: usize,
    error_rate: f64,
    thymic_q: f64,
}



#[derive(Default, Clone, Debug)]
struct GenerativeVJ{
    // Contains the distribution needed to generate the model
    d_v: DiscreteDistribution,
    d_j_given_v: Vec<DiscreteDistribution>,
    d_ins_vj: DiscreteDistribution,
    d_del_v_given_v: Vec<DiscreteDistribution>,
    d_del_j_given_j: Vec<DiscreteDistribution>,
    markov_vj: MarkovDNA,
}


#[derive(Default, Clone, Debug)]
pub struct ModelVJ{
    // Sequence information
    seg_vs: Vec<Gene>,
    seg_js: Vec<Gene>,

    // V/J nucleotides sequences trimmed at the CDR3 region (include F/W/C residues) with
    // the maximum number of reverse palindromic insertions appended.
    pub seg_vs_sanitized: Vec<Dna>, // match genomic_data.cutV_genomic_CDR3_seqs
    seg_js_sanitized: Vec<Dna>,

    // Probabilities of the different events
    pub p_v: Array1<f64>,
    pub p_j_given_v: Array2<f64>,
    pub p_ins_vj: Array1<f64>,
    pub p_del_v_given_v: Array2<f64>,
    pub p_del_j_given_j: Array2<f64>,
    gen: GenerativeVJ,
    markov_coefficients_vj: Array2<f64>,
    first_nt_bias_ins_vj: Array2<f64>,
    max_del_v: usize,
    max_del_j: usize,
    error_rate: f64,
    thymic_q: f64,
}


#[duplicate_item(model; [ModelVDJ]; [ModelVJ])]
impl model {
    pub fn recreate_full_sequence(&self, dna: Dna, v_index: usize, j_index: usize) -> (Dna, String, String) {
	// Re-create the full sequence of the variable region (with complete V/J gene, not just the CDR3)
	let mut seq: Dna = Dna::new();
	let vgene = self.seg_vs[v_index].clone();
	let jgene = self.seg_js[j_index].clone();
	seq.extend(&vgene.seq.extract_subsequence(0, vgene.cdr3_pos.unwrap()));
	seq.extend(&dna);
	seq.extend(&jgene.seq.extract_subsequence(jgene.cdr3_pos.unwrap()+3, jgene.seq.len()));
	(seq, vgene.name, jgene.name)
    }
}



impl ModelVDJ {

    pub fn load_model(path_params: &Path, path_marginals: &Path,
		      path_anchor_vgene: &Path, path_anchor_jgene: &Path
    ) -> Result<ModelVDJ, Box<dyn Error>> {
	let mut model: ModelVDJ = Default::default();
	let pm: ParserMarginals = ParserMarginals::parse(path_marginals)?;
	let mut pp: ParserParams = ParserParams::parse(path_params)?;
	pp.add_anchors_gene(path_anchor_vgene, "v_choice")?;
	pp.add_anchors_gene(path_anchor_jgene, "j_choice")?;

	model.seg_vs = pp.params.get("v_choice")
	    .ok_or("Error with unwrapping the Params data")?.clone().to_genes()?;
	model.seg_js = pp.params.get("j_choice")
	    .ok_or("Error with unwrapping the Params data")?.clone().to_genes()?;
	model.seg_ds = pp.params.get("d_gene")
	    .ok_or("Error with unwrapping the Params data")?.clone().to_genes()?;

	let mindelv = pp.params.get("v_3_del")
	    .ok_or("Invalid v_3_del")?.clone().to_numbers()?;
	model.max_del_v = (-mindelv.iter().min().ok_or("Empty v_3_del")?).try_into().map_err(|_e| "Invalid v_3_del")?;
	let mindelj = pp.params.get("j_5_del")
	    .ok_or("Invalid j_5_del")?.clone().to_numbers()?;
	model.max_del_j = (-mindelj.iter().min().ok_or("Empty j_5_del")?)
	    .try_into().map_err(|_e| "Invalid j_5_del")?;
	let mindel3 = pp.params.get("d_3_del")
	    .ok_or("Invalid d_3_del")?.clone().to_numbers()?;
	model.max_del_d3 = (-mindel3.iter().min().ok_or("Empty d_3_del")?)
	    .try_into().map_err(|_e| "Invalid d_3_del")?;
	let mindel5 = pp.params.get("d_5_del")
	    .ok_or("Invalid d_5_del")?.clone().to_numbers()?;
	model.max_del_d5 = (-mindel5.iter().min().ok_or("Empty d_5_del")?)
	    .try_into().map_err(|_e| "Invalid d_5_del")?;


	model.sanitize_genes()?;

	// Set the different probabilities for the model
	model.p_v = pm.marginals.get("v_choice").unwrap()
	    .probabilities.clone().into_dimensionality().unwrap();
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
	// TODO: check that the arrays of numbers are sorted and that they contain all the values
	model.p_ins_vd = pm.marginals.get("vd_ins").unwrap()
	    .probabilities.clone().into_dimensionality().unwrap();
	model.p_ins_dj = pm.marginals.get("dj_ins").unwrap()
	    .probabilities.clone().into_dimensionality().unwrap();
	model.p_del_v_given_v = pm.marginals.get("v_3_del").unwrap()
	    .probabilities.clone().into_dimensionality().unwrap().t().to_owned();
	model.p_del_j_given_j = pm.marginals.get("j_5_del").unwrap()
	    .probabilities.clone().into_dimensionality().unwrap().t().to_owned();
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
	model.markov_coefficients_vd = pm.marginals.get("vd_dinucl").unwrap()
	    .probabilities.clone().into_shape((4,4)).map_err(|_e| "Wrong size for vd_dinucl")?;
	model.markov_coefficients_dj = pm.marginals.get("dj_dinucl").unwrap()
	    .probabilities.clone().into_shape((4,4)).map_err(|_e| "Wrong size for dj_dinucl")?;

	// TODO: Need to deal with potential first nt bias

	// generative model
	model.initialize_generative_model()?;

	model.error_rate = pp.error_rate;
	model.thymic_q = 9.41; // TODO: deal with this
	Ok(model)
    }



    fn sanitize_genes(&mut self) -> Result<(), Box<dyn Error>> {
	// Trim the V/J nucleotides sequences at the CDR3 region (include F/W/C residues)
	// and append the maximum number of reverse palindromic insertions appended.

	self.seg_vs_sanitized = sanitize_v(self.seg_vs.clone(),
					   self.max_del_v);
	self.seg_js_sanitized = sanitize_j(self.seg_js.clone(),
					   self.max_del_j);
	self.seg_ds_sanitized = sanitize_d(self.seg_ds.clone(),
					    self.max_del_d5,
					   self.max_del_d3);
	Ok(())
    }


    fn initialize_generative_model(&mut self) -> Result<(), Box<dyn Error>>{
    	self.gen.d_v = DiscreteDistribution::new(self.p_v.to_vec())?;
    	self.gen.d_dj = DiscreteDistribution::new(self.p_dj.view().iter()
						  .cloned().collect())?;
    	self.gen.d_ins_vd = DiscreteDistribution::new(self.p_ins_vd.to_vec())?;
    	self.gen.d_ins_dj = DiscreteDistribution::new(self.p_ins_dj.to_vec())?;

    	self.gen.d_del_v_given_v = Vec::new();
	for row in self.p_del_v_given_v.axis_iter(Axis(1)) {
	    self.gen.d_del_v_given_v.push(DiscreteDistribution::new(row.to_vec())?);
	}
	self.gen.d_del_j_given_j = Vec::new();
	for row in self.p_del_j_given_j.axis_iter(Axis(1)) {
	    self.gen.d_del_j_given_j.push(DiscreteDistribution::new(row.to_vec())?);
	}

	self.gen.d_del_d3_del_d5 = Vec::new();
	for ii in 0..self.p_del_d3_del_d5.dim().2 {
	    let d3d5: Vec<f64> = self.p_del_d3_del_d5.slice(s![.., .., ii]).iter()
		.cloned().collect();
	    self.gen.d_del_d3_del_d5.push(DiscreteDistribution::new(d3d5)?);
	}

	self.gen.markov_vd = MarkovDNA::new(self.markov_coefficients_vd.t().to_owned(), None)?;
	self.gen.markov_dj = MarkovDNA::new(self.markov_coefficients_dj.t().to_owned(), None)?;
	Ok(())

    }


    pub fn generate<R: Rng>(&mut self, functional: bool, rng: &mut R) -> (Dna, Option<AminoAcid>, usize, usize){

	// loop until we find a valid sequence (if generating functional alone)
	loop{
	    let v_index: usize = self.gen.d_v.generate(rng);
	    let dj_index: usize = self.gen.d_dj.generate(rng);
	    let d_index: usize = dj_index / self.p_dj.dim().1;
	    let j_index: usize = dj_index % self.p_dj.dim().1;

	    let seq_v: &Dna = &self.seg_vs_sanitized[v_index];
	    let seq_d: &Dna = &self.seg_ds_sanitized[d_index];
	    let seq_j: &Dna = &self.seg_js_sanitized[j_index];

	    let del_v: usize = self.gen.d_del_v_given_v[v_index].generate(rng);
	    let del_d: usize = self.gen.d_del_d3_del_d5[d_index].generate(rng);
	    let del_d5: usize = del_d / self.p_del_d3_del_d5.dim().0;
	    let del_d3: usize = del_d % self.p_del_d3_del_d5.dim().0;
	    let del_j: usize = self.gen.d_del_j_given_j[j_index].generate(rng);

	    let ins_vd: usize = self.gen.d_ins_vd.generate(rng);
	    let ins_dj: usize = self.gen.d_ins_dj.generate(rng);

	    let out_of_frame = (seq_v.len() - del_v + seq_d.len()
			   - del_d5 - del_d3 + seq_j.len()
			   - del_j + ins_vd + ins_dj)%3 != 0;
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
		},
		None => {
		    if functional {continue;}
		    return (seq, None, v_index, j_index);

		}
	    }
	}
    }

}




impl ModelVJ {

    pub fn load_model(path_params: &Path, path_marginals: &Path,
		      path_anchor_vgene: &Path, path_anchor_jgene: &Path
    ) -> Result<ModelVJ, Box<dyn Error>> {
	let mut model: ModelVJ = Default::default();
	let pm: ParserMarginals = ParserMarginals::parse(path_marginals)?;
	let mut pp: ParserParams = ParserParams::parse(path_params)?;
	pp.add_anchors_gene(path_anchor_vgene, "v_choice")?;
	pp.add_anchors_gene(path_anchor_jgene, "j_choice")?;

	model.seg_vs = pp.params.get("v_choice")
	    .ok_or("Error with unwrapping the Params data")?.clone().to_genes()?;
	model.seg_js = pp.params.get("j_choice")
	    .ok_or("Error with unwrapping the Params data")?.clone().to_genes()?;

	let mindelv = pp.params.get("v_3_del")
	    .ok_or("Invalid v_3_del")?.clone().to_numbers()?;
	model.max_del_v = (-mindelv.iter().min().ok_or("Empty v_3_del")?)
	    .try_into().map_err(|_e| "Invalid v_3_del")?;
	let mindelj = pp.params.get("j_5_del")
	    .ok_or("Invalid j_5_del")?.clone().to_numbers()?;
	model.max_del_j = (-mindelj.iter().min().ok_or("Empty j_5_del")?)
	    .try_into().map_err(|_e| "Invalid j_5_del")?;

	model.sanitize_genes()?;

	// Set the different probabilities for the model
	model.p_v = pm.marginals.get("v_choice").unwrap()
	    .probabilities.clone().into_dimensionality().unwrap();
	model.p_j_given_v = pm.marginals.get("j_choice").unwrap()
	    .probabilities.clone().into_dimensionality().unwrap().t().to_owned();
	model.p_del_v_given_v = pm.marginals.get("v_3_del").unwrap()
	    .probabilities.clone().into_dimensionality().unwrap().t().to_owned();
	model.p_del_j_given_j = pm.marginals.get("j_5_del").unwrap()
	    .probabilities.clone().into_dimensionality().unwrap().t().to_owned();

	// TODO: check that the arrays of numbers are sorted and that they contain all the values
	model.p_ins_vj = pm.marginals.get("vj_ins").unwrap()
	    .probabilities.clone().into_dimensionality().unwrap();

	// Markov coefficients
	model.markov_coefficients_vj = pm.marginals.get("vj_dinucl").unwrap()
	    .probabilities.clone().into_shape((4,4)).map_err(|_e| "Wrong size for vj_dinucl")?;

	// TODO: Need to deal with potential first nt bias

	// generative model
	model.initialize_generative_model()?;

	model.error_rate = pp.error_rate;
	model.thymic_q = 9.41; // TODO: deal with this
	Ok(model)
    }



    fn sanitize_genes(&mut self) -> Result<(), Box<dyn Error>> {
	// Trim the V/J nucleotides sequences at the CDR3 region (include F/W/C residues)
	// and append the maximum number of reverse palindromic insertions appended.

	self.seg_vs_sanitized = sanitize_v(self.seg_vs.clone(),
					   self.max_del_v);
	self.seg_js_sanitized = sanitize_j(self.seg_js.clone(),
					   self.max_del_j);
	Ok(())
    }


    fn initialize_generative_model(&mut self) -> Result<(), Box<dyn Error>>{
    	self.gen.d_v = DiscreteDistribution::new(self.p_v.to_vec())?;
    	self.gen.d_ins_vj = DiscreteDistribution::new(self.p_ins_vj.to_vec())?;

	self.gen.d_j_given_v = Vec::new();
	for row in self.p_j_given_v.axis_iter(Axis(1)) {
	    self.gen.d_j_given_v.push(DiscreteDistribution::new(row.to_vec())?);
	}

    	self.gen.d_del_v_given_v = Vec::new();
	for row in self.p_del_v_given_v.axis_iter(Axis(1)) {
	    self.gen.d_del_v_given_v.push(DiscreteDistribution::new(row.to_vec())?);
	}
	self.gen.d_del_j_given_j = Vec::new();
	for row in self.p_del_j_given_j.axis_iter(Axis(1)) {
	    self.gen.d_del_j_given_j.push(DiscreteDistribution::new(row.to_vec())?);
	}

	self.gen.markov_vj = MarkovDNA::new(self.markov_coefficients_vj.t().to_owned(), None)?;
	Ok(())

    }


    pub fn generate<R: Rng>(&mut self, functional: bool, rng: &mut R) -> (Dna, Option<AminoAcid>, usize, usize){

	// loop until we find a valid sequence (if generating functional alone)
	loop{
	    let v_index: usize = self.gen.d_v.generate(rng);
	    let j_index: usize = self.gen.d_j_given_v[v_index].generate(rng);

	    let seq_v: &Dna = &self.seg_vs_sanitized[v_index];
	    let seq_j: &Dna = &self.seg_js_sanitized[j_index];

	    let del_v: usize = self.gen.d_del_v_given_v[v_index].generate(rng);
	    let del_j: usize = self.gen.d_del_j_given_j[j_index].generate(rng);

	    let ins_vj: usize = self.gen.d_ins_vj.generate(rng);

	    let out_of_frame = (seq_v.len() - del_v + seq_j.len()
			   - del_j + ins_vj)%3 != 0;
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
		},
		None => {
		    if functional {continue;}
		    return (seq, None, v_index, j_index);

		}
	    }
	}
    }
}



fn sanitize_v(genes: Vec<Gene>, max_del_v: usize) -> Vec<Dna> {
    // Add palindromic inserted nucleotides to germline V sequences and cut all
    // sequences to only keep their CDR3 parts
    let mut cut_genes = Vec::<Dna>::new();
    for g in genes {
	// some V-genes are not complete. They don't appear in the model, but we
	// can't ignore them // I need to change the way this is done...
	if g.cdr3_pos.unwrap() >= g.seq.len(){
	    cut_genes.push(Dna::new());
	    continue;
	}

	let mut gene_seq: Dna = g.seq;
	// Palindromic extension
	// ATGGTAC -> ATGGTACGTACC...
	let palindromic_extension = gene_seq.extract_subsequence(
	    gene_seq.len() - max_del_v, gene_seq.len())
	    .reverse_complement();
	gene_seq.seq.extend(palindromic_extension.seq);

	let cut_gene: Dna = Dna{seq:gene_seq.seq[g.cdr3_pos.unwrap()..].to_vec()};
	cut_genes.push(cut_gene);
    }
    cut_genes
}

fn sanitize_j(genes: Vec<Gene>, max_del_j: usize) -> Vec<Dna> {
    // Add palindromic inserted nucleotides to germline J sequences and cut all
    // sequences to only keep their CDR3 parts
    let mut cut_genes = Vec::<Dna>::new();
    for g in genes {
	let mut gene_seq: Dna = g.seq;
	// Palindromic extension
	// ATGGTAC -> ATGGTACGTACC...
	let palindromic_extension = gene_seq.extract_subsequence(
	    0, max_del_j)
	    .reverse_complement();
	gene_seq.seq = palindromic_extension.seq.into_iter()
	    .chain(gene_seq.seq.into_iter())
	    .collect();
	// for J, we want to also add the last CDR3 amino-acid (F/W)
	let cut_gene: Dna = Dna{seq:gene_seq.seq[..g.cdr3_pos.unwrap()+3+max_del_j].to_vec()};
	cut_genes.push(cut_gene);
    }
    cut_genes
}

fn sanitize_d(genes: Vec<Gene>, max_del_5_d: usize,  max_del_3_d: usize) -> Vec<Dna> {
    // Add palindromic inserted nucleotides to germline D sequences (both ends)
    let mut sanitized_genes = Vec::<Dna>::new();
    for g in genes {
	let mut gene_seq: Dna = g.seq;
	// Palindromic extension
	// ATGGTAC -> ATGGTACGTACC...
	let palindromic_extension_left = gene_seq.extract_subsequence(
	    0, max_del_5_d)
	    .reverse_complement();
	gene_seq.seq = palindromic_extension_left.seq.into_iter()
	    .chain(gene_seq.seq.into_iter())
	    .collect();
	let palindromic_extension_right = gene_seq.extract_subsequence(
	    gene_seq.len() - max_del_3_d, gene_seq.len())
	    .reverse_complement();
	gene_seq.seq.extend(palindromic_extension_right.seq);
	sanitized_genes.push(gene_seq);
    }
    sanitized_genes
}
