// Align the sequence to V & J genes

use crate::model::ModelVDJ;
use crate::utils::Gene;
use crate::utils_sequences::{differences, AlignmentParameters, Dna};

#[derive(Default, Clone, Debug)]
pub struct VJAlignment {
    // Structure containing the alignment between a V/J gene and the sequence
    // Note that the genes defined here include the palindromic insertion at the end

    // Example
    // gene (V):  ATACGATCATTGACAATCTGGAGATACGTA
    //                         ||||||\|||||\|\\\
    // sequence:               CAATCTAGAGATTCCAATCTAGAGATTCA
    // start_gene -------------^ 13
    // end_gene   ------------------------------^ 30
    // start_seq               0
    // end_seq                 -----------------^ 17
    // errors[u] contains the number of errors left if u nucleotides are removed
    // (starting from the end of the V-gene, or the beginning of the J gene)
    // the length of this vector is the maximum number of v/j deletion.
    // so here [5,4,3,2,2,1,1,1,1,1,1,0,0,...]
    pub index: usize, // index of the gene in the model
    pub start_seq: usize,
    pub end_seq: usize,
    pub start_gene: usize,
    pub end_gene: usize,
    pub errors: Vec<usize>,
}

#[derive(Default, Clone, Debug)]
pub struct DAlignment {
    // Structure containing the alignment between a D gene and the sequence
    // Similarly to VJaligment the gene include the palindromic insertion
    // d-gene  :         DDDDDDDDDDDD
    // sequence:     SSSSSSSSSSSSSSSSSSSSS
    // pos     ----------^ 4
    // error_left contains the number of errors in the alignment starting from the left
    // error_right contains the number of errors in the alignment starting from the right
    // For example:
    // DDDDDDDDDDD
    // SXSSSXSSXSX
    // errors_left:  [4,4,3,3,3,3,2,2,2,1,1]
    // errors_right: [4,3,3,2,2,2,1,1,1,1,0]
    pub index: usize,
    pub len_d: usize, // length of the D gene
    pub pos: usize,   // begining of the D-gene in the sequence (can't be < 0)
    pub errors_left: Vec<usize>,
    pub errors_right: Vec<usize>,
}

#[derive(Default, Clone, Debug)]
pub struct SequenceVDJ {
    pub sequence: Dna,
    // subset of reasonable v_genes, j_genes
    pub v_genes: Vec<VJAlignment>,
    pub j_genes: Vec<VJAlignment>,
    pub d_genes: Vec<DAlignment>,
}

impl SequenceVDJ {
    pub fn align_sequence(
        seq: Dna,
        model: &ModelVDJ,
        align_params: &AlignmentParameters,
    ) -> SequenceVDJ {
        let mut seq = SequenceVDJ {
            sequence: seq.clone(),
            v_genes: align_all_vgenes(&seq, model, align_params),
            j_genes: align_all_jgenes(&seq, model, align_params),
            d_genes: Vec::new(),
        };
        // roughly estimate bounds for the position of d
        let left_bound = seq
            .v_genes
            .iter()
            .map(|v| v.end_seq - model.max_del_v)
            .min();
        let right_bound = seq
            .j_genes
            .iter()
            .map(|v| j.start_seq + model.max_del_j)
            .max();

        // initialize all the d genes positions
        seq.d_genes = align_all_dgenes(&seq, model, left_bound, right_bound, align_params);
        seq
    }

    pub fn get_insertions_vd_dj(
        &self,
        v: &VJAlignment,
        delv: usize,
        d: &DAlignment,
        deld3: usize,
        deld5: usize,
        j: &VJAlignment,
        delj: usize,
    ) -> (Dna, Dna) {
        // seq         :          SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
        // V-gene      : VVVVVVVVVVVVVVVVVVVV
        // del V-gene  :                   xx
        // D-gene      :                                 DDDDDDDDDDD
        // del D5      :                                 xxx
        // del D3      :                                        xxxx
        // J-gene      :                                                          JJJJJJJJJJJ
        // del J5      :                                                          x
        // insertions  :                   IIIIIIIIIIIIIIIII    IIIIIIIIIIIIIIIIIII

        // the first insertion starts at v + len(v) - delv, then
        // go to d + deld5
        // the second insertion starts at d + len(d) - deld3 then
        // go to j + delj
        let dgene_len = d.gene.seq_with_pal.as_ref().unwrap().len();
        let start_insvd = v.end_gene - delv;
        let end_insvd = d.pos + deld3;
        let start_insdj = d.pos + dgene_len - deld5;
        let end_insdj = j.start_gene + dgene_len + delj;

        let insvd = self.sequence.extract_subsequence(start_insvd, end_insvd);
        let insdj = self.sequence.extract_subsequence(start_insdj, end_insdj);
        (insvd, insdj)
    }
}

fn align_all_vgenes(
    seq: &Dna,
    model: &ModelVDJ,
    align_params: &AlignmentParameters,
) -> Vec<VJAlignment> {
    let mut v_genes: Vec<VJAlignment> = Vec::new();
    for (indexv, v) in model.seg_vs.iter().enumerate() {
        let palv = v.seq_with_pal.as_ref().unwrap();
        let alignment = Dna::align_left_right(&palv, &seq, align_params);
        if align_params.valid_v_alignment(&alignment) {
            v_genes.push(VJAlignment {
                index: indexv,
                start_gene: alignment.xstart,
                end_gene: alignment.xend,
                start_seq: alignment.ystart,
                end_seq: alignment.yend,
                errors: differences_remaining(
                    seq.seq[alignment.ystart..alignment.yend].iter().rev(),
                    palv.seq[alignment.xstart..alignment.xend].iter().rev(),
                    model.max_delv,
                ),
            });
        }
    }
    v_genes
}

fn align_all_jgenes(
    seq: &Dna,
    model: &ModelVDJ,
    align_params: &AlignmentParameters,
) -> Vec<VJAlignment> {
    let mut j_aligns: Vec<VJAlignment> = Vec::new();
    for (indexj, j) in model.seg_js.iter().enumerate() {
        let palj = j.seq_with_pal.as_ref().unwrap();
        let alignment = Dna::align_left_right(&seq, &palj, align_params);
        if align_params.valid_j_alignment(&alignment) {
            j_aligns.push(VJAlignment {
                index: indexj,
                start_gene: alignment.ystart,
                end_gene: alignment.yend,
                start_seq: alignment.xstart,
                end_seq: alignment.xend,
                errors: differences_remaining(
                    seq.seq[alignment.xstart..].iter(),
                    palj.seq.iter(),
                    max_del_j,
                ),
            });
        }
    }
    j_aligns
}

fn align_all_dgenes(
    seq: &Dna,
    model: &ModelVDJ,
    limit_5side: usize,
    limit_3side: usize,
    align_params: &AlignmentParameters,
) -> Vec<DAlignment> {
    // For each D gene, we test all the potential positions of insertion
    // between limit_5side and limit_3side.
    // Note that the "D-gene" include all the palindromic end
    // The D-gene need to be completely covered by the read
    //    DDDDDDDDDDDDDD
    // SSSSSSSSSSSSSSSSSSSSSS

    let mut daligns: Vec<DAlignment> = Vec::new();
    for (indexd, d) in model.seg_ds.iter().enumerate() {
        let dpal = d.seq_with_pal.as_ref().unwrap();
        for pos in limit_5side..limit_3side - d.seq.len() {
            let errors_left = differences_remaining(
                seq.seq[pos..pos + dpal.len()].iter().rev(),
                dpal.iter().rev(),
                d.len(),
            );
            let errors_right =
                differences_remaining(seq.seq[pos..pos + dpal.len()].iter(), dpal.iter(), d.len());

            daligns.push(DAlignment {
                index: indexd,
                pos: pos,
                len_d: dpal.len(),
                errors_left: errors_left,
                errors_right: errors_right,
            });
        }
    }
    daligns
}
