use crate::sequence::utils::{difference_as_i64, differences_remaining};
use crate::sequence::Dna;
use crate::sequence::{AlignmentParameters, DAlignment, VJAlignment};
use crate::vj::{Event, Model};
#[cfg(all(feature = "py_binds", feature = "py_o3"))]
use pyo3::*;

#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug)]
pub struct Sequence {
    pub sequence: Dna,
    // subset of reasonable v_genes, j_genes
    pub v_genes: Vec<VJAlignment>,
    pub j_genes: Vec<VJAlignment>,
    pub d_genes: Vec<DAlignment>,
}

impl Sequence {
    pub fn get_insertions_vj(&self, e: &Event) -> Dna {
        // seq         :          SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
        // V-gene      : VVVVVVVVVVVVVVVVVVVV
        // del V-gene  :                   xx
        // J-gene      :                                                          JJJJJJJJJJJ
        // del J5      :                                                          x
        // insertion   :                   IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII

        // the first insertion starts at v + len(v) - delv, then
        // go to d + deld5
        // the second insertion starts at d + len(d) - deld3 then
        // go to j + delj
        // in the situation where there's two much deletion and some insertion are undefined
        // return N for the unknown sites, for ex
        // seq         : SSSS
        // J-gene      :   JJJJJJJJJ
        // del J       :   xxxxx
        // V           : V
        // would return for insVJ: SSNNNNN

        // this one can potentially be negative
        let start_insvj = difference_as_i64(e.v.end_seq, e.delv);
        let end_insvj = e.j.start_seq + e.delj;
        let insvj = self
            .sequence
            .extract_padded_subsequence(start_insvj, end_insvj as i64);
        insvj
    }
}

pub fn align_all_vgenes(
    seq: &Dna,
    model: &Model,
    align_params: &AlignmentParameters,
) -> Vec<VJAlignment> {
    let mut v_genes: Vec<VJAlignment> = Vec::new();
    for (indexv, v) in model.seg_vs.iter().enumerate() {
        let palv = v.seq_with_pal.as_ref().unwrap();
        let alignment = Dna::align_left_right(palv, seq, align_params);
        // println!(
        //     "{}",
        //     alignment.pretty(palv.seq.as_slice(), seq.seq.as_slice(), 200)
        // );
        // println!("{:?}", alignment.score);

        if align_params.valid_v_alignment(&alignment) {
            v_genes.push(VJAlignment {
                index: indexv,
                start_gene: alignment.xstart,
                end_gene: alignment.xend,
                start_seq: alignment.ystart,
                end_seq: alignment.yend,
                errors: differences_remaining(
                    seq.seq[alignment.ystart..alignment.yend]
                        .iter()
                        .rev()
                        .copied(),
                    palv.seq[alignment.xstart..alignment.xend]
                        .iter()
                        .rev()
                        .copied(),
                    (model.range_del_v.1 - model.range_del_v.0) as usize,
                ),
            });
        }
    }
    v_genes
}

pub fn align_all_jgenes(
    seq: &Dna,
    model: &Model,
    align_params: &AlignmentParameters,
) -> Vec<VJAlignment> {
    let mut j_aligns: Vec<VJAlignment> = Vec::new();
    for (indexj, j) in model.seg_js.iter().enumerate() {
        let palj = j.seq_with_pal.as_ref().unwrap();
        let alignment = Dna::align_left_right(seq, palj, align_params);
        // println!(
        //     "{}",
        //     alignment.pretty(seq.seq.as_slice(), palj.seq.as_slice(), 200)
        // );
        // println!("{:?}", alignment.score);
        if align_params.valid_j_alignment(&alignment) {
            j_aligns.push(VJAlignment {
                index: indexj,
                start_gene: alignment.ystart,
                end_gene: alignment.yend,
                start_seq: alignment.xstart,
                end_seq: alignment.xend,
                errors: differences_remaining(
                    seq.seq[alignment.xstart..].iter().copied(),
                    palj.seq.iter().copied(),
                    (model.range_del_j.1 - model.range_del_j.0) as usize,
                ),
            });
        }
    }
    j_aligns
}
