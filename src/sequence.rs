// Align the sequence to V & J genes

use crate::model::ModelVDJ;
use crate::utils_sequences::{difference_as_i64, differences_remaining, AlignmentParameters, Dna};
use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use std::cmp;

pub struct EventVDJ<'a> {
    pub v: &'a VJAlignment,
    pub j: &'a VJAlignment,
    pub d: &'a DAlignment,
    pub delv: usize,
    pub delj: usize,
    pub deld3: usize,
    pub deld5: usize,
}

#[pyclass(name = "EventVDJ", get_all, set_all)]
#[derive(Default, Clone, Debug)]
pub struct StaticEventVDJ {
    pub v_index: usize,
    pub v_start_gene: usize, // start of the sequence in the gene
    pub delv: usize,
    pub j_index: usize,
    pub j_start_seq: usize, // start of the sequence in the sequence
    pub delj: usize,
    pub d_index: usize,
    pub d_start_seq: usize,
    pub deld3: usize,
    pub deld5: usize,
    pub insvd: Dna,
    pub insdj: Dna,
}

impl EventVDJ<'_> {
    pub fn to_static(&self, insvd: Dna, insdj: Dna) -> StaticEventVDJ {
        StaticEventVDJ {
            v_index: self.v.index,
            v_start_gene: self.v.start_gene,
            delv: self.delv,
            j_index: self.j.index,
            j_start_seq: self.j.start_seq,
            delj: self.delj,
            d_index: self.d.index,
            d_start_seq: self.d.pos,
            deld3: self.deld3,
            deld5: self.deld5,
            insvd,
            insdj,
        }
    }
}

#[pyclass(get_all, set_all)]
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

#[pymethods]
impl VJAlignment {
    pub fn nb_errors(&self, del: usize) -> usize {
        if del >= self.errors.len() {
            return match self.errors.last() {
                None => 0,
                Some(l) => *l,
            };
        }
        self.errors[del]
    }
}

#[pyclass(get_all, set_all)]
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
    // "pos" represents the start of the sequence *with palindromes added*
    pub index: usize,
    len_d: usize,   // length of the D gene (with palindromic inserts)
    pub pos: usize, // begining of the D-gene in the sequence (can't be < 0)
    pub errors_left: Vec<usize>,
    pub errors_right: Vec<usize>,
}

#[pyclass(get_all, set_all)]
#[derive(Default, Clone, Debug)]
pub struct SequenceVDJ {
    pub sequence: Dna,
    // subset of reasonable v_genes, j_genes
    pub v_genes: Vec<VJAlignment>,
    pub j_genes: Vec<VJAlignment>,
    pub d_genes: Vec<DAlignment>,
}

#[pymethods]
impl DAlignment {
    pub fn nb_errors(&self, deld3: usize, deld5: usize) -> usize {
        self.errors_left[deld5] + self.errors_right[deld3]
    }
    pub fn len(&self) -> usize {
        self.len_d
    }
    pub fn is_empty(&self) -> bool {
        self.len_d == 0
    }
}

#[pymethods]
impl SequenceVDJ {
    #[staticmethod]
    pub fn align_sequence(
        dna_seq: Dna,
        model: &ModelVDJ,
        align_params: &AlignmentParameters,
    ) -> Result<SequenceVDJ> {
        let mut seq = SequenceVDJ {
            sequence: dna_seq.clone(),
            v_genes: align_all_vgenes(&dna_seq, model, align_params),
            j_genes: align_all_jgenes(&dna_seq, model, align_params),
            d_genes: Vec::new(),
        };

        // if we don't have v genes, don't try inferring the d gene
        if (seq.v_genes.is_empty()) | (seq.j_genes.is_empty()) {
            return Ok(seq);
        }

        // roughly estimate bounds for the position of d
        // TODO: not great, improve on that
        let left_bound = seq
            .v_genes
            .iter()
            .map(|v| {
                if v.end_seq > (model.range_del_v.1 as usize) {
                    v.end_seq - (model.range_del_v.1 as usize)
                } else {
                    0
                }
            })
            .min()
            .ok_or(anyhow!("Error in the definition of the D gene bounds"))?;
        let right_bound = seq
            .j_genes
            .iter()
            .map(|j| cmp::min(j.start_seq + (model.range_del_j.1 as usize), dna_seq.len()))
            .max()
            .ok_or(anyhow!("Error in the definition of the D gene bounds"))?;

        // initialize all the d genes positions
        seq.d_genes = align_all_dgenes(&dna_seq, model, left_bound, right_bound, align_params);
        Ok(seq)
    }
}

impl SequenceVDJ {
    pub fn get_insertions_vd_dj(&self, e: &EventVDJ) -> (Dna, Dna) {
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

fn align_all_vgenes(
    seq: &Dna,
    model: &ModelVDJ,
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

fn align_all_jgenes(
    seq: &Dna,
    model: &ModelVDJ,
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
