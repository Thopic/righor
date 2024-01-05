use crate::sequence::utils::{count_differences, difference_as_i64};
use crate::sequence::Dna;
use crate::sequence::{AlignmentParameters, DAlignment, VJAlignment};
use crate::vj::{Event, Model};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::*;

#[cfg_attr(
    all(feature = "py_binds", feature = "pyo3"),
    pyclass(get_all, set_all)
)]
#[derive(Default, Clone, Debug)]
pub struct Sequence {
    pub sequence: Dna,
    // subset of reasonable v_genes, j_genes
    pub v_genes: Vec<VJAlignment>,
    pub j_genes: Vec<VJAlignment>,
    pub d_genes: Vec<DAlignment>,
}

impl Sequence {
    pub fn best_v_alignment(&self) -> Option<VJAlignment> {
        self.v_genes.clone().into_iter().max_by_key(|m| m.score)
    }

    pub fn best_j_alignment(&self) -> Option<VJAlignment> {
        self.j_genes.clone().into_iter().max_by_key(|m| m.score)
    }

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
        if align_params.valid_v_alignment(&alignment) {
            // println!(
            //     "{}",
            //     alignment.pretty(palv.seq.as_slice(), seq.seq.as_slice(), 200)
            // );
            // println!("V: {:?}", alignment.score);
            let max_del_v = model.p_del_v_given_v.dim().0;

            let mut errors = vec![0; max_del_v];
            for del_v in 0..max_del_v {
                if del_v <= palv.len() && del_v <= alignment.yend - alignment.ystart {
                    errors[del_v] = count_differences(
                        &seq.seq[alignment.ystart..alignment.yend - del_v],
                        &palv.seq[alignment.xstart..alignment.xend - del_v],
                    );
                }
            }

            v_genes.push(VJAlignment {
                index: indexv,
                start_gene: alignment.xstart,
                end_gene: alignment.xend,
                start_seq: alignment.ystart,
                end_seq: alignment.yend,
                errors,
                length: palv.len(),
                score: alignment.score,
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
        if align_params.valid_j_alignment(&alignment) {
            // println!(
            //     "{}",
            //     alignment.pretty(seq.seq.as_slice(), palj.seq.as_slice(), 200)
            // );
            // println!("J: {:?}", alignment.score);
            let max_del_j = model.p_del_j_given_j.dim().0;
            let mut errors = vec![0; max_del_j];
            for del_j in 0..max_del_j {
                if del_j <= palj.len() && del_j <= alignment.yend - alignment.ystart {
                    errors[del_j] = count_differences(
                        &seq.seq[del_j + alignment.xstart..alignment.xend],
                        &palj.seq[del_j + alignment.ystart..alignment.yend],
                    );
                }
            }

            j_aligns.push(VJAlignment {
                index: indexj,
                start_gene: alignment.ystart,
                end_gene: alignment.yend,
                start_seq: alignment.xstart,
                end_seq: alignment.xend,
                errors,
                length: palj.len(),
                score: alignment.score,
            });
        }
    }
    j_aligns
}

pub fn display_j_alignment(
    seq: &Dna,
    j_al: &VJAlignment,
    model: &Model,
    align_params: &AlignmentParameters,
) -> String {
    let j = model.seg_js[j_al.index].clone();
    let palj = j.seq_with_pal.as_ref().unwrap();
    let alignment = Dna::align_left_right(seq, palj, align_params);
    alignment.pretty(seq.seq.as_slice(), palj.seq.as_slice(), 100)
}

pub fn display_v_alignment(
    seq: &Dna,
    v_al: &VJAlignment,
    model: &Model,
    align_params: &AlignmentParameters,
) -> String {
    let v = model.seg_vs[v_al.index].clone();
    let palv = v.seq_with_pal.as_ref().unwrap();
    let alignment = Dna::align_left_right(palv, seq, align_params);
    alignment.pretty(palv.seq.as_slice(), seq.seq.as_slice(), 100)
}
