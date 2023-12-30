use crate::sequence::{DAlignment, Dna, VJAlignment};
use anyhow::{anyhow, Result};
#[cfg(all(feature = "py_binds", feature = "py_o3"))]
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

#[cfg_attr(
    all(feature = "py_binds", feature = "py_o3"),
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
    pub d_index: usize,
    pub d_start_seq: usize,
    pub deld3: usize,
    pub deld5: usize,
    pub insvd: Dna,
    pub insdj: Dna,
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
        })
    }

    pub fn default() -> Self {
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
