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
    // errors contains the position of each error in the alignment
    // (starting from the end of the V-gene, or the beginning of the J gene)
    // so here [0,1,2,4,7,10]
    pub gene: Gene,
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
    pub gene: Gene,
    pub pos: usize,         // begining of the D-gene in the sequence (can't be < 0)
    pub errors: Vec<usize>, // position of the error in the D gene
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
        SequenceVDJ {
            sequence: seq.clone(),
            v_genes: align_all_vgenes(&seq, model, align_params),
            j_genes: align_all_jgenes(&seq, model, align_params),
            d_genes: Vec::new(),
        }
    }

    fn get_insertions_vd_dj(
        &self,
        v: usize,
        delv: usize,
        d: usize,
        deld3: usize,
        deld5: usize,
        j: usize,
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
        let dgene_len = self.d_genes[d].gene.seq_with_pal.as_ref().unwrap().len();
        let start_insvd = self.v_genes[v].end_gene - delv;
        let end_insvd = self.d_genes[d].pos + deld3;
        let start_insdj = self.d_genes[d].pos + dgene_len - deld5;
        let end_insdj = self.j_genes[j].start_gene + dgene_len + delj;

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
    for v in &model.seg_vs {
        let palv = v.seq_with_pal.as_ref().unwrap();
        let alignment = Dna::align_left_right(&palv, &seq, align_params);
        if align_params.valid_v_alignment(&alignment) {
            v_genes.push(VJAlignment {
                gene: v.clone(),
                start_gene: alignment.xstart,
                end_gene: alignment.xend,
                start_seq: alignment.ystart,
                end_seq: alignment.yend,
                errors: differences(
                    seq.seq[alignment.ystart..alignment.yend].iter().rev(),
                    palv.seq[alignment.xstart..alignment.xend].iter().rev(),
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
    for j in &model.seg_js {
        let palj = j.seq_with_pal.as_ref().unwrap();
        let alignment = Dna::align_left_right(&seq, &palj, align_params);
        if align_params.valid_j_alignment(&alignment) {
            j_aligns.push(VJAlignment {
                gene: j.clone(),
                start_gene: alignment.ystart,
                end_gene: alignment.yend,
                start_seq: alignment.xstart,
                end_seq: alignment.xend,
                errors: differences(seq.seq[alignment.xstart..].iter(), palj.seq.iter()),
            });
        }
    }
    j_aligns
}

fn align_all_dgenes(
    seq: &Dna,
    model: ModelVDJ,
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
    for d in &model.seg_ds {
        let dpal = d.seq_with_pal.as_ref().unwrap();
        for pos in limit_5side..limit_3side - d.seq.len() {
            let errors = differences(seq.seq[pos..pos + d.seq.len()].iter(), dpal.seq.iter());
            if errors.len() > align_params.max_error_d {
                continue;
            }
            daligns.push(DAlignment {
                gene: d.clone(),
                pos: pos,
                errors: errors,
            });
        }
    }
    daligns
}
