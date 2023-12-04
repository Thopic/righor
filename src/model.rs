
#[derive(Default, Clone, Debug)]
struct GenerativeVDJ{
    // Contains the distribution needed to
    d_v: DiscreteDistribution,
    d_dj: DiscreteDistribution,
    d_ins_vd: DiscreteDistribution,
    d_ins_dj: DiscreteDistribution,
    d_del_v_given_v: Vec[DiscreteDistribution],
    d_del_j_given_j: Vec[DiscreteDistribution],
    d_del_d3_del_d5: Vec[DiscreteDistribution],
    markov_vd: MarkovDna,
    markov_dj: MarkovDna,

}


#[derive(Default, Clone, Debug)]
struct ModelVDJ{
    verbose: bool,
    random_generator: rand::rngs::StdRng,

    // Sequence information
    seg_vs: Vec<Gene>,
    seg_js: Vec<Gene>,
    seg_ds: Vec<Gene>,

    // V/J nucleotides sequences trimmed at the CDR3 region (include F/W/C residues) with
    // the maximum number of reverse palindromic insertions appended.
    seg_vs_sanitized: Vec<Dna>, // match genomic_data.cutV_genomic_CDR3_seqs
    seg_js_sanitized: Vec<Dna>,
    seg_ds_sanitized: Vec<Dna>,

    // Probabilities of the different events
    p_v: Array1<f64>,
    p_dj: Array2<f64>,
    p_ins_vd: Array1<f64>,
    p_ins_dj: Array1<f64>,
    p_del_v_given_v: Array2<f64>,
    p_del_j_given_j: Array2<f64>,
    p_del_d3_del_d5_given_d: Array2<f64>
    gen: GenerativeVDJ,
    markov_coefficients_vd: Array2<f64>,
    markov_coefficients_dj: Array2<f64>,
    first_nt_bias_ins_vd: Array2<f64>,
    first_nt_bias_ins_dj: Array2<f64>,
    error_rate: f64,
    thymic_q: f64,
}

impl ModelVDJ {

    pub fn load_model(path_params: &Path, path_marginals: &Path,
		      path_anchor_vgene: &Path, path_anchor_jgene: &Path
    ) -> Result<ModelVDJ, String> {
	let mut model: ModelVDJ = Default::default();
	let pm: ParserMarginals = ParserMarginals::parse(path_marginals);
	let pp: ParserParams = ParserParams::parse(path_params);
	pp.add_anchor_genes(path_anchor_vgene, "v_choice");
	pp.add_anchor_genes(path_anchor_jgene, "j_choice");



	model.sanitize_genes(&pp);

	// Set the different probabilities for the model
	model.p_v = pm.marginals.get("v_choice");
	// For the joint probability P(D, J), we just multiply
	// P(J) and P(D|J)
	// model.p_dj[d, j] = pj[j] * pd[j, d]
	let pd = pm.marginals.get("d_gene").probabilities;
	let pj = pm.marginals.get("j_choice").probabilities;
	let [jdim, ddim] = pd.dim();
	model.p_dj = Array2::<f64>::zeros((ddim, jdim));
	for dd in 0..ddim {
	    for jj in 0..jdim {
		model.p_dj[dd, jj] = pd[jj, dd] * pj[j]
	    }
	}
	// TODO: check that the arrays of numbers are sorted and that they contain all the values
	model.p_ins_vd = pm.marginals.get("vd_ins").probabilities;
	model.p_ins_dj = pm.marginals.get("dj_ins").probabilities;
	model.p_del_v_given_v = pm.marginals.get("del_v_3").probabilities.T;
	model.p_del_j_given_j = pm.marginals.get("del_j_5").probabilities.T;
	// compute the joint probability P(delD3, delD5 | D)
	// P(delD3, delD5 | D) = P(delD3 | delD5, D) * P(delD5 | D)
	let pdeld3 = pm.marginals.get("del_d_3").probabilities; // P(deld3| delD5, D)
	let pdeld5 = pm.marginals.get("del_d_5").probabilities; // P(deld5| D)
	let [ddim, d5dim, d3dim] = pdeld3.dim();
	model.p_del_d3_del_d5 = Array3::<f64>::zeros((d3dim, d5dim, ddim));
	for dd in 0..ddim {
	    for d5 in 0..d5dim {
		for d3 in 0..d3dim {
		    model.p_del_d3_del_d5[d3, d5, d] = pdeld3[d, d5, d3] * pdeld3[d, d5]
		}
	    }
	}

	// Markov coefficients
	model.markov_coefficients_vd = pm.marginals.get("vd_dinucl")
	    .probabilities.into_shape((4,4));
	model.markov_coefficients_dj = pm.marginals.get("dj_dinucl")
	    .probabilities.into_shape((4,4));

	// TODO: Need to deal with potential first nt bias


	model.error_rate = pp.error_rate;
	model.thymic_q = 9.41; // TODO: deal with this
    }



    fn sanitize_genes(&mut self, pp: &ParserParams){
	// Trim the V/J nucleotides sequences at the CDR3 region (include F/W/C residues)
	// and append the maximum number of reverse palindromic insertions appended.

	let max_delv_palindrome = - match pp.params.get("v_3_del")? {
	    Event::Numbers(num) => {nums.into_iter().min().ok_or("Invalid v_3_del")},
	    _ => Err("Invalid parsing, error with v_3_del")}?;
	let max_delj_palindrome = - match pp.params.get("j_5_del")? {
	    Event::Numbers(num) => {nums.into_iter().min().ok_or("Invalid j_5_del")},
	    _ => Err("Invalid parsing, error with j_5_del")}?;
	let max_deld3_palindrome = - match pp.params.get("d_3_del")? {
	    Event::Numbers(num) => {nums.into_iter().min().ok_or("Invalid d_3_del")},
	    _ => Err("Invalid parsing, error with d_3_del")}?;
	let max_deld5_palindrome = - match pp.params.get("d_5_del")? {
	    Event::Numbers(num) => {nums.into_iter().min().ok_or("Invalid d_5_del")},
	    _ => Err("Invalid parsing, error with d_5_del")}?;

	self.seg_vs_sanitized = sanitize_v(model.seg_vs, max_delv_palindrome);
	self.seg_js_sanitized = sanitize_j(model.seg_js, max_delj_palindrome);
	self.seg_ds_sanitized = sanitize_d(model.seg_ds,
					    max_deld5_palindrome,
					    max_deld3_palindrome);
    }


    // fn initialize_generative_model(){
    // 	// define the distributions
    // 	model.d_v  = DiscreteDistribution(pm.marginals.get("v_choice")
    // 					  .probabilities
    // 					  .iter()
    // 					  .cloned()
    // 					  .collect());

    // 	model.d_dj = DiscreteDistribution(pj.dot(&pd).iter().cloned().collect());


    // 	model.d_ins_vd = DiscreteDistribution(pm.marginals.get("vd_ins")
    // 					      .probabilities
    // 					      .iter()
    // 					      .cloned()
    // 					      .collect());

    // 	model.d_ins_dj = DiscreteDistribution(pm.marginals.get("dj_ins")
    // 					      .probabilities
    // 					      .iter()
    // 					      .cloned()
    // 					      .collect());

    // 	model.d_del_v_given_v = DiscreteDistribution(pm.marginals.get("v_3_del")
    // 						     .probabilities.iter().cloned().collect()



    // }

}






fn sanitize_v(genes: Vec<Gene>, max_del_v: usize) -> Vec<Dna> {
    // Add palindromic inserted nucleotides to germline V sequences and cut all
    // sequences to only keep their CDR3 parts
    let mut cut_genes = Vec<Dna>::new();
    for g in genes {
	let mut gene_seq: Dna = g.seq;
	// Palindromic extension
	// ATGGTAC -> ATGGTACGTACC...
	let palindromic_extension = gene_seq.extract_subsequence(
	    gene_seq.len() - max_del_v, gene_seq.len())
	    .reverse_complement();
	gene_seq.seq.extend(palindromic_extension.seq);
	let cut_gene: Dna = gene_seq.seq.seq[g.cdr3_pos..];
	cut_genes.push(cut_gene);
    }
    cut_genes
}

fn sanitize_j(genes: Vec<Gene>, max_del_j: usize) -> Vec<Dna> {
    // Add palindromic inserted nucleotides to germline J sequences and cut all
    // sequences to only keep their CDR3 parts
    let mut cut_genes = Vec<Dna>::new();
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
	let cut_gene: Dna = gene_seq.seq.seq[..g.cdr3_pos+max_del_j];
	cut_genes.push(cut_gene);
    }
    cut_genes
}

fn sanitize_d(genes: Vec<Gene>, max_del_5_d: usize,  max_del_3_d: usize) -> Vec<Dna> {
    // Add palindromic inserted nucleotides to germline D sequences (both ends)
    let mut sanitized_genes = Vec<Dna>::new();
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
