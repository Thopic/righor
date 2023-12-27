use crate::sequence::utils::{difference_as_i64, differences_remaining, AlignmentParameters};
use crate::sequence::{DAlignment, Dna, VJAlignment};
use crate::vdj::{Event, Model};

use pyo3::*;

#[pyclass(get_all, set_all)]
#[derive(Default, Clone, Debug)]
pub struct Sequence {
    pub sequence: Dna,
    // subset of reasonable v_genes, j_genes
    pub v_genes: Vec<VJAlignment>,
    pub j_genes: Vec<VJAlignment>,
    pub d_genes: Vec<DAlignment>,
}

impl Sequence {
    pub fn get_insertions_vd_dj(&self, e: &Event) -> (Dna, Dna) {
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
        // in the situation where there's two much deletion and some insertion are undefined
        // return N for the unknown sites, for ex
        // seq         : SSSS
        // J-gene      :   JJJJJJJJJ
        // del J       :   xxxxx
        // D           : D
        // would return for insDJ: SSNNNNN

        // this one can potentially be negative
        let start_insvd = difference_as_i64(e.v.end_seq, e.delv);
        let end_insvd = e.d.pos + e.deld5;
        let start_insdj = e.d.pos + e.d.len() - e.deld3;
        let end_insdj = e.j.start_seq + e.delj;
        let insvd = self
            .sequence
            .extract_padded_subsequence(start_insvd, end_insvd as i64);
        let insdj = self
            .sequence
            .extract_padded_subsequence(start_insdj as i64, end_insdj as i64);
        (insvd, insdj)
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
        if align_params.valid_j_alignment(&alignment) {
            // println!(
            //     "{}",
            //     alignment.pretty(seq.seq.as_slice(), palj.seq.as_slice(), 200)
            // );
            // println!("J: {:?}", alignment.score);

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

pub fn align_all_dgenes(
    seq: &Dna,
    model: &Model,
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
        for pos in limit_5side..=limit_3side - dpal.len() {
            let errors_left = differences_remaining(
                seq.seq[pos..pos + dpal.len()].iter().rev().copied(),
                dpal.seq.iter().rev().copied(),
                dpal.len(),
            );
            let errors_right = differences_remaining(
                seq.seq[pos..pos + dpal.len()].iter().copied(),
                dpal.seq.iter().copied(),
                dpal.len(),
            );

            if (!errors_left.is_empty()) & (errors_left[0] > align_params.max_error_d) {
                continue;
            }

            daligns.push(DAlignment {
                index: indexd,
                pos,
                len_d: dpal.len(),
                errors_left,
                errors_right,
            });
        }
    }
    daligns
}
