use crate::sequence::{Dna, VJAlignment};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

pub struct Event<'a> {
    pub v: &'a VJAlignment,
    pub j: &'a VJAlignment,
    pub delv: usize,
    pub delj: usize,
}

#[cfg_attr(
    all(feature = "py_binds", feature = "pyo3"),
    pyclass(name = "Event", get_all, set_all)
)]
#[derive(Default, Clone, Debug)]
pub struct StaticEvent {
    pub v_index: usize,
    pub v_start_gene: usize, // start of the sequence in the gene
    pub delv: usize,
    pub j_index: usize,
    pub j_start_seq: usize, // start of the sequence in the sequence
    pub delj: usize,
    pub insvj: Dna,
}

impl Event<'_> {
    pub fn to_static(&self, insvj: Dna) -> StaticEvent {
        StaticEvent {
            v_index: self.v.index,
            v_start_gene: self.v.start_gene,
            delv: self.delv,
            j_index: self.j.index,
            j_start_seq: self.j.start_seq,
            delj: self.delj,
            insvj,
        }
    }
}
