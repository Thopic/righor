use crate::shared::sequence::SequenceType;
use crate::shared::DnaLike;
use crate::shared::{utils::difference_as_i64, AlignmentParameters, DAlignment, Dna, VJAlignment};
use crate::vdj::{Event, Model};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use std::sync::Arc;

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Clone, Debug)]
pub struct Sequence {
    pub sequence: DnaLike,
    // subset of reasonable v_genes, j_genes
    pub v_genes: Vec<VJAlignment>,
    pub j_genes: Vec<VJAlignment>,
    pub d_genes: Vec<DAlignment>,
    pub valid_alignment: bool,
    pub sequence_type: SequenceType,
}

impl Sequence {
    pub fn get_subsequence(&self, start: i64, end: i64) -> DnaLike {
        self.sequence.extract_padded_subsequence(start, end)
    }

    pub fn best_v_alignment(&self) -> Option<VJAlignment> {
        self.v_genes.clone().into_iter().max_by_key(|m| m.score)
    }

    pub fn best_j_alignment(&self) -> Option<VJAlignment> {
        self.j_genes.clone().into_iter().max_by_key(|m| m.score)
    }

    pub fn get_specific_dgene(&self, d_idx: usize) -> Vec<DAlignment> {
        self.d_genes
            .clone()
            .into_iter()
            .filter(|d| d.index == d_idx)
            .collect()
    }

    pub fn get_insertions_vd_dj(&self, e: &Event) -> (DnaLike, DnaLike) {
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

        let v = e.v.unwrap();
        let d = e.d.unwrap();
        let j = e.j.unwrap();

        let start_insvd = difference_as_i64(v.end_seq, e.delv);
        let end_insvd = d.pos + e.deld5 as i64;
        let start_insdj = d.pos + d.len() as i64 - e.deld3 as i64;
        let end_insdj = j.start_seq as i64 - j.start_gene as i64 + e.delj as i64;
        let insvd = self
            .sequence
            .extract_padded_subsequence(start_insvd, end_insvd);
        let insdj = self
            .sequence
            .extract_padded_subsequence(start_insdj, end_insdj);
        (insvd, insdj)
    }
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
    alignment.pretty(seq.seq.as_slice(), palj.seq.as_slice(), 80)
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
    // Sadly alignment.pretty is bugged exactly for this use case,
    // I'm waiting for them to correct this
    // https://github.com/rust-bio/rust-bio-types/issues/47
    alignment.pretty(palv.seq.as_slice(), seq.seq.as_slice(), 80)
}

pub fn align_all_vgenes(
    seq: &DnaLike,
    model: &Model,
    align_params: &AlignmentParameters,
) -> Vec<VJAlignment> {
    let mut v_genes: Vec<VJAlignment> = Vec::new();
    for (indexv, v) in model.seg_vs.iter().enumerate() {
        let palv = v.seq_with_pal.as_ref().unwrap();
        let Some(alignment) = DnaLike::v_alignment(palv, seq.clone(), align_params) else {
            continue;
        };

        let mut v_alignment = VJAlignment {
            index: indexv,
            start_gene: alignment.xstart,
            end_gene: alignment.xend,
            start_seq: alignment.ystart,
            end_seq: alignment.yend,
            score: alignment.score,
            max_del: Some(model.p_del_v_given_v.shape()[0]),
            gene_sequence: palv.clone(),
            sequence_type: seq.sequence_type(),
            ..Default::default()
        };

        v_alignment.precompute_errors_v(seq);

        v_genes.push(v_alignment);
    }
    v_genes
}

pub fn align_all_jgenes(
    seq: &DnaLike,
    model: &Model,
    align_params: &AlignmentParameters,
) -> Vec<VJAlignment> {
    let mut j_aligns: Vec<VJAlignment> = Vec::new();
    for (indexj, j) in model.seg_js.iter().enumerate() {
        let palj = j.seq_with_pal.clone().unwrap();
        let alignment =
            DnaLike::align_left_right(seq.clone(), DnaLike::from_dna(palj.clone()), align_params);
        if align_params.valid_j_alignment(&alignment) {
            let mut j_al = VJAlignment {
                index: indexj,
                start_gene: alignment.ystart,
                end_gene: alignment.yend,
                start_seq: alignment.xstart,
                end_seq: alignment.xend,
                score: alignment.score,
                max_del: Some(model.p_del_j_given_j.dim().0),
                gene_sequence: palj.clone(),
                sequence_type: seq.sequence_type(),
                ..Default::default()
            };
            j_al.precompute_errors_j(seq);
            j_aligns.push(j_al);
        }
    }
    j_aligns
}

pub fn align_all_dgenes(
    seq: &DnaLike,
    model: &Model,
    limit_5side: i64,
    limit_3side: i64,
    align_params: &AlignmentParameters,
) -> Vec<DAlignment> {
    // For each D gene, we test all the potential positions of insertion
    // between limit_5side and limit_3side.
    // Note that the "D-gene" include all the palindromic end
    // The D-gene need to be completely covered by the read
    //    DDDDDDDDDDDDDD
    // SSSSSSSSSSSSSSSSSSSSSS

    let seq_ref = Arc::new(seq.clone());
    let mut daligns: Vec<DAlignment> = Vec::new();
    for (indexd, d) in model.seg_ds.iter().enumerate() {
        let dpal = d.seq_with_pal.as_ref().unwrap();
        let dpal_ref = Arc::new(d.seq_with_pal.clone().unwrap());
        for pos in limit_5side..=limit_3side - dpal.len() as i64 {
            if pos + (dpal.len() as i64) < 0 {
                continue;
            }
            if seq
                .extract_padded_subsequence(pos, pos + dpal.len() as i64)
                .count_differences(&dpal)
                > align_params.max_error_d
            {
                continue;
            }

            daligns.push(DAlignment {
                index: indexd,
                pos,
                len_d: dpal.len(),
                dseq: dpal_ref.clone(),
                sequence: seq_ref.clone(),
                sequence_type: seq.sequence_type(),
            });
        }
    }
    daligns
}
