// Align the sequence to V & J genes
use crate::sequence::Dna;
use crate::vdj::model::Model as ModelVDJ;
#[cfg(all(feature = "py_binds", feature = "py_o3"))]
use pyo3::prelude::*;

#[cfg_attr(
    all(feature = "py_binds", feature = "py_o3"),
    pyclass(get_all, set_all)
)]
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
    pub index: usize, // index of the gene in the model
    pub start_seq: usize,
    pub end_seq: usize,
    pub start_gene: usize,
    pub end_gene: usize,
    pub errors: Vec<usize>,
    pub score: i32,
}

#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pymethods)]
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

#[cfg_attr(
    all(feature = "py_binds", feature = "py_o3"),
    pyclass(get_all, set_all)
)]
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
    pub len_d: usize,            // length of the D gene (with palindromic inserts)
    pub pos: usize,              // begining of the D-gene in the sequence (can't be < 0)
    pub errors: Vec<Vec<usize>>, // errors[deld5][deld3] = #number of errors left with deld5, deld3
}

#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pymethods)]
impl DAlignment {
    pub fn nb_errors(&self, deld5: usize, deld3: usize) -> usize {
        self.errors[deld5][deld3]
    }
    pub fn len(&self) -> usize {
        self.len_d
    }
    pub fn is_empty(&self) -> bool {
        self.len_d == 0
    }

    pub fn display(&self, sequence: &Dna, model: &ModelVDJ) -> String {
        let mut line1 = "".to_string();
        let mut line2 = "".to_string();
        let mut line3 = "".to_string();
        let dna_sequence = sequence.seq.clone();
        let dna_dgene = model.seg_ds[self.index].seq_with_pal.clone().unwrap().seq;
        for ii in 0..sequence.len() {
            line1 += &(dna_sequence[ii] as char).to_string();
            if (ii < self.pos) || (ii >= self.pos + self.len_d) {
                line2 += " ";
                line3 += " ";
            } else {
                line3 += &(dna_dgene[ii - self.pos] as char).to_string();
                if dna_dgene[ii - self.pos] != dna_sequence[ii] {
                    line2 += "\\";
                } else {
                    line2 += "|";
                }
            }
        }
        format!("{line1}\n{line2}\n{line3}\n")
    }

    pub fn total_error(&self) -> usize {
        self.errors[0][0]
    }
}
