use crate::shared::sequence::Dna;
use crate::vj::Model;
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct StaticEvent {
    pub v_index: usize,
    pub v_start_gene: usize, // start of the sequence in the V gene
    pub delv: usize,
    pub j_index: usize,
    pub j_start_seq: i64, // start of the palindromic J gene (with all dels) in the sequence
    pub delj: usize,
    pub insvj: Dna,
    pub errors: Vec<(usize, u8)>,
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl StaticEvent {
    pub fn __repr__(&self) -> String {
        format!(
            "StaticEvent(\n\
		 nb. del. on V3: {},\n\
		 nb. del. on J5: {},\n\
		 V-J insertions: {},\n\
		 errors: {})",
            self.delv,
            self.delj,
            self.insvj.get_string(),
            if self.errors.is_empty() {
                "None".to_string()
            } else {
                self.errors
                    .iter()
                    .map(|(num, byte)| format!("S{}{}", num, *byte as char))
                    .collect::<Vec<_>>()
                    .join("")
            },
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

        // add errors
        for (ii, nuc) in &self.errors {
            seq.seq[*ii] = *nuc;
        }

        seq
    }

    pub fn extract_cdr3(&self, full_sequence: &Dna, m: &Model) -> Dna {
        let vg = &m.seg_vs[self.v_index];
        let jg = &m.seg_js[self.j_index];
        let mut end_cdr3 = full_sequence.len() - jg.seq.len() + jg.cdr3_pos.unwrap() + 3;
        if vg.cdr3_pos.unwrap() > end_cdr3 {
            end_cdr3 = vg.cdr3_pos.unwrap(); // if we cut too much we return the empty sequence
        }
        full_sequence.extract_subsequence(vg.cdr3_pos.unwrap(), end_cdr3)
    }

    pub fn to_cdr3(&self, m: &Model) -> Dna {
        let full_sequence = self.to_sequence(m);
        self.extract_cdr3(&full_sequence, m)
    }
}
