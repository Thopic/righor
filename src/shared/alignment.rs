use crate::shared::errors::MAX_NB_ERRORS;
use crate::shared::nucleotides_inv;
use crate::shared::sequence::Dna;
use crate::shared::sequence::SequenceType;
use crate::shared::DnaLike;
use crate::vdj::model::Model as ModelVDJ;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use std::sync::Arc;

//use super::sequence::DnaLikeEnum;

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug)]
pub struct ErrorAlignment {
    pub nb_errors: usize,
    pub sequence_length: usize,
}

pub struct ErrorVAlignment<'a> {
    pub val: &'a VJAlignment,
    pub del: usize,
}

pub struct ErrorJAlignment<'a> {
    pub jal: &'a VJAlignment,
    pub del: usize,
}

pub struct ErrorDAlignment<'a> {
    pub dal: &'a DAlignment,
    pub deld3: usize,
    pub deld5: usize,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
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
    // score is the score of the alignment according to the alignment process
    pub index: usize,      // index of the gene in the model
    pub start_seq: usize,  // this is the start of the alignment in the sequence indexing
    pub end_seq: usize,    // end of the alignment in the sequence indexing
    pub start_gene: usize, // start of the alignment in the gene indexing
    pub end_gene: usize,   // end of the alignment in the gene indexing
    pub errors: Vec<usize>,
    pub errors_extended: Option<Vec<[usize; 16]>>,
    pub score: i32,
    pub max_del: Option<usize>,
    pub gene_sequence: Dna, // v/j gene sequence (with pal insertions)
    pub sequence_type: SequenceType,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl VJAlignment {
    /// the first nucleotide of "gene" when the sequence start at j_start
    /// elements are removed at the start
    pub fn get_first_nucleotide(&self, del: usize) -> usize {
        nucleotides_inv(self.gene_sequence.seq[self.start_gene + del])
    }

    // the last nucleotide of "gene" when delv elements are removed at the end
    pub fn get_last_nucleotide(&self, del: usize) -> usize {
        nucleotides_inv(self.gene_sequence.seq[self.end_gene - del - 1])
    }

    pub fn nb_errors(&self, del: usize) -> usize {
        if del >= self.errors.len() {
            return match self.errors.last() {
                None => 0,
                Some(l) => *l,
            };
        }
        self.errors[del]
    }

    pub fn valid_extended_j(&self, del: usize) -> Vec<usize> {
        debug_assert!(del < self.errors_extended.as_ref().unwrap().len());
        self.errors_extended.as_ref().unwrap()[del]
            .iter()
            .enumerate()
            .filter(|&(_i, &value)| value == 0)
            .map(|(i, _value)| i)
            .collect()
    }

    pub fn precompute_errors_v(&mut self, seq: &DnaLike) {
        self.errors = vec![0; self.max_del.unwrap()];
        for del_v in 0..self.errors.len() {
            if self.end_seq > del_v + seq.len() {
                // large number (ugly hack, but shouldn't create issues)
                self.errors[del_v] = MAX_NB_ERRORS;
            } else if self.start_seq + del_v <= self.end_seq
                && self.start_gene + del_v <= self.end_gene
                && self.end_seq <= del_v + seq.len()
                && self.end_gene <= del_v + self.gene_sequence.len()
            {
                self.errors[del_v] = seq
                    .extract_subsequence(self.start_seq, self.end_seq - del_v)
                    .count_differences(
                        &self
                            .gene_sequence
                            .extract_subsequence(self.start_gene, self.end_gene - del_v),
                    );
            }
        }
        // no problems here
        self.errors_extended = None;
    }

    pub fn precompute_errors_j(&mut self, seq: &DnaLike) {
        self.errors = vec![0; self.max_del.unwrap()];
        let mut errors_extended = vec![[0; 16]; self.max_del.unwrap()];

        for del_j in 0..self.errors.len() {
            // if palJ[delJ:] start before the beginning of the sequence
            if (del_j as i64 - self.start_gene as i64 + self.start_seq as i64) < 0 {
                self.errors[del_j] = MAX_NB_ERRORS;
                if seq.is_protein() {
                    errors_extended[del_j] = [MAX_NB_ERRORS; 16];
                }
            }
            // if palJ[delJ:] does still overlap with the CDR3 sequence
            else if del_j <= self.end_gene {
                let cut_seq = seq.extract_padded_subsequence(
                    del_j as i64 - self.start_gene as i64 + self.start_seq as i64,
                    self.end_seq as i64,
                );
                let cut_gene = self.gene_sequence.extract_subsequence(del_j, self.end_gene);

                self.errors[del_j] = cut_seq.count_differences(&cut_gene);
                // TODO: Simplify this
                if seq.is_protein() {
                    let cut_seq_plus_idx = seq.extract_padded_subsequence(
                        del_j as i64 - self.start_gene as i64 + self.start_seq as i64 - 2,
                        self.end_seq as i64,
                    );

                    for idx in 0..16 {
                        let mut gene_plus_idx = Dna::from_matrix_idx(idx);
                        gene_plus_idx.extend(&cut_gene);
                        errors_extended[del_j][idx] =
                            cut_seq_plus_idx.count_differences(&gene_plus_idx);
                    }
                }
            }
            self.errors_extended = Some(errors_extended.clone());
        }
    }

    pub fn length_with_deletion(&self, del_left: usize, del_right: usize) -> usize {
        let del = if del_right > 0 {
            // J case
            if self.start_gene > self.start_seq {
                if del_right > (self.start_gene - self.start_seq) {
                    del_right - (self.start_gene - self.start_seq)
                } else {
                    0
                }
            } else {
                del_right
            }
        } else {
            // V case
            // if the V gene is longer than the sequence aligned
            if self.gene_sequence.len() > self.end_gene {
                if del_left > (self.gene_sequence.len() - self.end_gene) {
                    del_left - (self.gene_sequence.len() - self.end_gene)
                } else {
                    0
                }
            } else {
                del_left
            }
        };
        if del > self.end_gene - self.start_gene {
            0
        } else {
            self.end_gene - self.start_gene - del
        }
    }

    pub fn errors(&self, del_left: usize, del_right: usize) -> ErrorAlignment {
        debug_assert!(del_left == 0 || del_right == 0);
        ErrorAlignment {
            nb_errors: self.nb_errors(del_left + del_right),
            sequence_length: self.length_with_deletion(del_left, del_right),
        }
    }

    pub fn estimated_error_rate(&self, max_del_left: usize, max_del_right: usize) -> f64 {
        (self.nb_errors(max_del_left + max_del_right) as f64)
            / (self.length_with_deletion(max_del_left, max_del_right) as f64)
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
#[derive(Clone, Debug)]
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
    pub len_d: usize, // length of the D gene (with palindromic inserts)
    pub pos: i64,     // begining of the D-gene in the sequence

    pub dseq: Arc<Dna>,         // the sequence of the D gene
    pub sequence: Arc<DnaLike>, // the complete sequence aligned
    pub sequence_type: SequenceType,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl DAlignment {
    pub fn nb_errors(&self, deld5: usize, deld3: usize) -> usize {
        if deld5 + deld3 > self.len_d {
            return 0;
        }

        if self.pos + (deld5 as i64) < 0 {
            return MAX_NB_ERRORS;
        }

        self.sequence
            .extract_subsequence(
                (self.pos + deld5 as i64) as usize,
                (self.pos + self.len_d as i64 - deld3 as i64) as usize,
            )
            .count_differences(
                &self
                    .dseq
                    .extract_subsequence(deld5, self.dseq.len() - deld3),
            )
    }
    pub fn length_with_deletion(&self, deld5: usize, deld3: usize) -> usize {
        self.len() - deld5 - deld3
    }

    pub fn valid_extremities(&self, deld5: usize, deld3: usize) -> Vec<(usize, usize)> {
        debug_assert!(deld5 + deld3 <= self.len());
        let cut_d = self
            .dseq
            .extract_subsequence(deld5, self.dseq.len() - deld3);
        let cut_seq = self.sequence.extract_padded_subsequence(
            self.pos + deld5 as i64 - 2,
            self.pos + self.len_d as i64 - deld3 as i64,
        );

        let mut vec = vec![];

        for idx_left in 0..16 {
            let mut seq = Dna::from_matrix_idx(idx_left);
            seq.extend(&cut_d);
            if cut_seq.count_differences(&seq) == 0 {
                let idx_right = seq
                    .extract_subsequence(seq.len() - 2, seq.len())
                    .to_matrix_idx();
                debug_assert!(idx_right.len() == 1);
                vec.push((idx_left, idx_right[0]));
            }
        }
        vec
    }

    pub fn errors(&self, deld5: usize, deld3: usize) -> ErrorAlignment {
        ErrorAlignment {
            nb_errors: self.nb_errors(deld5, deld3),
            sequence_length: self.length_with_deletion(deld5, deld3),
        }
    }

    pub fn len(&self) -> usize {
        self.len_d
    }
    pub fn is_empty(&self) -> bool {
        self.len_d == 0
    }

    // /// Given an alignment d and the number of deletion on both side
    // /// return all possible extremities in the matrix format
    // /// Left extremities is the two nucleotides at the start of the D gene
    // /// Right extremities the two nucleotides at the end of the D gene
    // pub fn extremities(&self, deld5: usize, deld3: usize) -> Vec<(usize, usize)> {
    //     // default case
    //     //  DDDDDDDDDDDDDDD
    //     //  xx             oo
    //     //start idx     end idx

    //     // edge cases:
    //     // len(D_with_del) == 1
    //     //  D
    //     //  xx
    //     //   oo

    //     // len(D_with_del) == 0
    //     // xx
    //     // oo

    //     let left_side = self
    //         .dseq
    //         .extract_padded_subsequence(deld5, deld5 + 2)
    //         .to_matrix_idx();
    //     let right_side = self
    //         .sequence
    //         .extract_padded_subsequence(
    //             self.pos + self.len_d as i64 - deld3 as i64,
    //             self.pos + self.len_d as i64 - deld3 as i64 + 2,
    //         )
    //         .to_matrix_idx();

    //     if self.length_with_deletion(deld5, deld3) >= 2 {
    //         // left side and right side are independant
    //         return itertools.product(left_side, right_side).collect();
    //     } else if self.length_with_deletion(deld5, deld3) == 1 {
    //         let result = vec![];
    //         for (left, right) in itertools.product(left_side, right_side) {
    //             if left % 4 == right / 4 {
    //                 result.push((left, right))
    //             }
    //         }
    //         return result;
    //     } else {
    //         // if the D gene is completely removed
    //         let result = vec![];
    //         for (left, right) in itertools.product(left_side, right_side) {
    //             if left == right {
    //                 result.push((left, right))
    //             }
    //         }
    //         return result;
    //     }
    // }
}

#[cfg(feature = "py_binds")]
#[pymethods]
impl DAlignment {
    #[getter]
    pub fn get_index(&self) -> usize {
        self.index
    }

    #[getter]
    pub fn get_pos(&self) -> i64 {
        self.pos
    }
}

impl DAlignment {
    pub fn display(&self, sequence: &Dna, model: &ModelVDJ) -> String {
        if self.pos < 0 {
            unimplemented!("Working on it. Later.");
        }

        let mut line1 = "".to_string();
        let mut line2 = "".to_string();
        let mut line3 = "".to_string();
        let dna_sequence = sequence.seq.clone();
        let dna_dgene = model.seg_ds[self.index].seq_with_pal.clone().unwrap().seq;
        for ii in 0..sequence.len() {
            line1 += &(dna_sequence[ii] as char).to_string();
            if (ii < self.pos as usize) || (ii >= self.pos as usize + self.len_d) {
                line2 += " ";
                line3 += " ";
            } else {
                line3 += &(dna_dgene[ii - self.pos as usize] as char).to_string();
                if dna_dgene[ii - self.pos as usize] != dna_sequence[ii] {
                    line2 += "\\";
                } else {
                    line2 += "|";
                }
            }
        }
        format!("{line1}\n{line2}\n{line3}\n")
    }
}
