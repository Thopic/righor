use crate::sequence::{DAlignment, Dna, VJAlignment};
use crate::vdj::Model;
use anyhow::{anyhow, Result};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

pub struct Event<'a> {
    pub v: Option<&'a VJAlignment>,
    pub j: Option<&'a VJAlignment>,
    pub d: Option<&'a DAlignment>,
    pub delv: usize,
    pub delj: usize,
    pub deld3: usize,
    pub deld5: usize,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct StaticEvent {
    pub v_index: usize,
    pub v_start_gene: usize, // start of the sequence in the V gene
    pub delv: usize,
    pub j_index: usize,
    pub j_start_seq: usize, // start of the palindromic J gene (with all dels) in the sequence
    pub delj: usize,
    pub d_index: usize,
    pub d_start_seq: usize,
    pub deld3: usize,
    pub deld5: usize,
    pub insvd: Dna,
    pub insdj: Dna,
    pub errors: Vec<(usize, u8)>,
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl StaticEvent {
    fn __repr__(&self) -> String {
        format!(
            "StaticEvent(\n\
		 nb. del. on V3: {},\n\
		 nb. del. on D5: {},\n\
		 nb. del. on D3: {},\n\
		 nb. del. on J5: {},\n\
		 V-D insertions: {},\n\
		 D-J insertions: {})",
            self.delv,
            self.deld5,
            self.deld3,
            self.delj,
            self.insvd.get_string(),
            self.insdj.get_string()
        )
    }
}

impl StaticEvent {
    pub fn to_sequence(&self, m: &Model) -> Dna {
        let seq_v: &Dna = m.seg_vs[self.v_index].seq_with_pal.as_ref().unwrap();
        let seq_j: &Dna = m.seg_js[self.j_index].seq_with_pal.as_ref().unwrap();
        let seq_d: &Dna = m.seg_ds[self.d_index].seq_with_pal.as_ref().unwrap();
        let mut seq: Dna = Dna::new();
        seq.extend(&seq_v.extract_subsequence(0, seq_v.len() - self.delv));
        seq.extend(&self.insvd);
        seq.extend(&seq_d.extract_subsequence(self.deld5, seq_d.len() - self.deld3));
        seq.extend(&self.insdj);
        seq.extend(&seq_j.extract_subsequence(self.delj, seq_j.len()));

        // add errors
        for (ii, nuc) in &self.errors {
            seq.seq[*ii] = *nuc
        }

        seq
    }

    pub fn extract_cdr3(&self, full_sequence: &Dna, m: &Model) -> Dna {
        let vg = &m.seg_vs[self.v_index];
        let jg = &m.seg_js[self.j_index];
        let end_cdr3 = full_sequence.len() - jg.seq.len() + jg.cdr3_pos.unwrap() + 3;
        full_sequence.extract_subsequence(vg.cdr3_pos.unwrap(), end_cdr3)
    }

    pub fn to_cdr3(&self, m: &Model) -> Dna {
        let full_sequence = self.to_sequence(m);
        self.extract_cdr3(&full_sequence, m)
        // let seq_v_cdr3: &Dna = &m.seg_vs_sanitized[self.v_index];
        // let seq_j_cdr3: &Dna = &m.seg_js_sanitized[self.j_index];
        // let seq_d: &Dna = m.seg_ds[self.d_index].seq_with_pal.as_ref().unwrap();
        // let mut seq: Dna = Dna::new();

        // if self.delv > seq_v_cdr3.len() {
        //     // this should go to None, CDR3 is badly defined
        //     return None;
        // }
        // if self.delj > seq_j_cdr3.len() {
        //     return None;
        // }

        // // println!(
        // //     "{:?}",
        // //     (
        // //         seq_v_cdr3
        // //             .extract_subsequence(0, seq_v_cdr3.len() - self.delv)
        // //             .get_string(),
        // //         self.insvd.get_string(),
        // //         seq_d
        // //             .extract_subsequence(self.deld5, seq_d.len() - self.deld3)
        // //             .get_string(),
        // //         self.insdj.get_string(),
        // //         seq_j_cdr3
        // //             .extract_subsequence(self.delj, seq_j_cdr3.len())
        // //             .get_string()
        // //     )
        // // );

        // seq.extend(&seq_v_cdr3.extract_subsequence(0, seq_v_cdr3.len() - self.delv));
        // seq.extend(&self.insvd);
        // seq.extend(&seq_d.extract_subsequence(self.deld5, seq_d.len() - self.deld3));
        // seq.extend(&self.insdj);
        // seq.extend(&seq_j_cdr3.extract_subsequence(self.delj, seq_j_cdr3.len()));

        // Some(seq)
    }
}

impl Event<'_> {
    pub fn to_static(&self, insvd: Dna, insdj: Dna) -> Result<StaticEvent> {
        Ok(StaticEvent {
            v_index: self
                .v
                .ok_or(anyhow!("Can't move that event to static"))?
                .index,
            v_start_gene: self
                .v
                .ok_or(anyhow!("Can't move that event to static"))?
                .start_gene,
            delv: self.delv,
            j_index: self
                .j
                .ok_or(anyhow!("Can't move that event to static"))?
                .index,
            j_start_seq: self
                .j
                .ok_or(anyhow!("Can't move that event to static"))?
                .start_seq,
            delj: self.delj,
            d_index: self
                .d
                .ok_or(anyhow!("Can't move that event to static"))?
                .index,
            d_start_seq: self
                .d
                .ok_or(anyhow!("Can't move that event to static"))?
                .pos,
            deld3: self.deld3,
            deld5: self.deld5,
            insvd,
            insdj,
            ..Default::default()
        })
    }
}
impl Default for Event<'_> {
    fn default() -> Self {
        Event {
            v: None,
            j: None,
            d: None,
            delv: 0,
            delj: 0,
            deld3: 0,
            deld5: 0,
        }
    }
}
