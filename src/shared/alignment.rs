use crate::shared::errors::MAX_NB_ERRORS;
use crate::shared::nucleotides_inv;
use crate::shared::Dna;
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
    pub score: i32,
    pub max_del_v: Option<usize>,
    pub gene_sequence: Dna, // v/j gene sequence
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pymethods)]
impl VJAlignment {
    // the first nucleotide of "gene" when the sequence start at j_start
    // elements are removed at the start
    pub fn get_first_nucleotide(&self, del: usize) -> usize {
        // println!(
        //     "{} {} {}",
        //     self.gene_sequence.to_string(),
        //     self.start_gene,
        //     del
        // );
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

    pub fn length_with_deletion(&self, del: usize) -> usize {
        // just return the aligned part length (that matches the seq)
        if self.end_seq <= self.start_seq + del {
            return 0;
        }
        self.end_seq - self.start_seq - del
    }

    pub fn errors(&self, del: usize) -> ErrorAlignment {
        ErrorAlignment {
            nb_errors: self.nb_errors(del),
            sequence_length: self.length_with_deletion(del),
        }
    }

    pub fn estimated_error_rate(&self, max_del: usize) -> f64 {
        return (self.nb_errors(max_del) as f64) / (self.length_with_deletion(max_del) as f64);
    }

    // pub fn allowed_indices_end(delv: usize) {
    //     nucleotides_inv(self.gene_sequence.seq[self.end_gene - 1])
    // }
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

    pub fn gene_with_deletion(&self, deld5: usize, deld3: usize) -> DnaLike {
        let dgene = self
            .dseq
            .extract_subsequence(deld5, self.dseq.len() - deld3);
        let start_d_seq = (self.pos + deld5 as i64) as usize;
        let end_d_seq = (self.pos + self.len_d as i64 - deld3 as i64) as usize;

        // complete dgene with the alignment to get the right codons
        let mut extended_seq = self
            .sequence
            .extract_subsequence(start_d_seq - 2, start_d_seq);
        extended_seq.extend(dgene.into());
        extended_seq.extend(
            self.sequence
                .extract_subsequence(end_d_seq, end_d_seq + 2)
                .into(),
        );
        extended_seq.extract_subsequence(2, extended_seq.len() - 2)
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
