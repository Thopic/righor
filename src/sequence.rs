


// This module needs to:
// - read the sequence in memory, and infer plausible V / J genes
// - infer potential V/J genes mutations (? => probably on the python side)

struct Sequence {
    sequence: Dna,
    // subset of reasonable v_genes, j_genes
    v_genes: Vec<(usize, usize)>,
    j_genes: Vec<(usize, usize)>,
}


impl Sequence {

    fn align_sequence(seq: Dna, model: Model) -> Sequence {


    }

}



fn align_all_vgenes(seq: Dna, model: Model, align_params: AlignmentParameters) -> Vec<(usize, usize)> {
    for v in model.seq_vs {
	let alignment = Dna::align_left_right(&v.seq, &seq, &align_params);
	if alignment.score > align_params.min_score_v {

	}

    }

}

fn align_all_jgenes(seq: Dna, model: Model) -> Vec<(usize, usize)> {

}

fn align_vgene(seq: Dna, vgene: Gene) -> Option<Vec<usize>> {



}

fn align_jgene(seq: Dna, model: Model) -> Option<Vec<usize>> {

}
