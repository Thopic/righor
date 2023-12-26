use crate::sequence::{DAlignment, Dna, VJAlignment};
use pyo3::prelude::*;

pub struct Event<'a> {
    pub v: &'a VJAlignment,
    pub j: &'a VJAlignment,
    pub d: &'a DAlignment,
    pub delv: usize,
    pub delj: usize,
    pub deld3: usize,
    pub deld5: usize,
}

#[pyclass(name = "Event", get_all, set_all)]
#[derive(Default, Clone, Debug)]
pub struct StaticEvent {
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

impl Event<'_> {
    pub fn to_static(&self, insvd: Dna, insdj: Dna) -> StaticEvent {
        StaticEvent {
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
