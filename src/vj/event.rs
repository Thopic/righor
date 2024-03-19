use crate::sequence::Dna;
use crate::vdj::Model;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct StaticEvent {
    pub v_index: usize,
    pub v_start_gene: usize, // start of the sequence in the V gene
    pub delv: usize,
    pub j_index: usize,
    pub j_start_seq: usize, // start of the palindromic J gene (with all dels) in the sequence
    pub delj: usize,
    pub insvj: Dna,
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl StaticEvent {
    fn __repr__(&self) -> String {
        format!(
            "StaticEvent(\n\
		 nb. del. on V3: {},\n\
		 nb. del. on J5: {},\n\
		 V-J insertions: {})",
            self.delv,
            self.delj,
            self.insvj.get_string(),
        )
    }
}

impl StaticEvent {
    pub fn to_sequence(&self, m: &Model) -> Dna {
        let seq_v: &Dna = m.seg_vs[self.v_index].seq_with_pal.as_ref().unwrap();
        let seq_j: &Dna = m.seg_js[self.j_index].seq_with_pal.as_ref().unwrap();
        let mut seq: Dna = Dna::new();
        seq.extend(&seq_v.extract_subsequence(0, seq_v.len() - self.delv));
        seq.extend(&self.insvj);
        seq.extend(&seq_j.extract_subsequence(self.delj, seq_j.len()));
        seq
    }
    pub fn to_cdr3(&self, m: &Model) -> Dna {
        let seq_v_cdr3: &Dna = &m.seg_vs_sanitized[self.v_index];
        let seq_j_cdr3: &Dna = &m.seg_js_sanitized[self.j_index];
        let mut seq: Dna = Dna::new();

        seq.extend(&seq_v_cdr3.extract_subsequence(0, seq_v_cdr3.len() - self.delv));
        seq.extend(&self.insvj);
        seq.extend(&seq_j_cdr3.extract_subsequence(self.delj, seq_j_cdr3.len()));
        seq
    }
}
