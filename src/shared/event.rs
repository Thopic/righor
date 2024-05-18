use crate::shared::Dna;
use crate::shared::Model;
use crate::vdj::StaticEvent as VDJStaticEvent;
use crate::vj::StaticEvent as VJStaticEvent;
use anyhow::{anyhow, Result};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StaticEvent {
    VDJ(VDJStaticEvent),
    VJ(VJStaticEvent),
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pyclass(name = "StaticEvent")]
pub struct PyStaticEvent {
    pub s: StaticEvent,
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl PyStaticEvent {
    fn __repr__(&self) -> String {
        match &self.s {
            StaticEvent::VDJ(x) => x.__repr__(),
            StaticEvent::VJ(x) => x.__repr__(),
        }
    }

    #[getter]
    fn get_v_index(&self) -> PyResult<usize> {
        Ok(match &self.s {
            StaticEvent::VDJ(x) => x.v_index,
            StaticEvent::VJ(x) => x.v_index,
        })
    }

    #[getter]
    fn get_v_start_gene(&self) -> PyResult<usize> {
        Ok(match &self.s {
            StaticEvent::VDJ(x) => x.v_start_gene,
            StaticEvent::VJ(x) => x.v_start_gene,
        })
    }

    #[getter]
    fn get_delv(&self) -> PyResult<usize> {
        Ok(match &self.s {
            StaticEvent::VDJ(x) => x.delv,
            StaticEvent::VJ(x) => x.delv,
        })
    }

    #[getter]
    fn get_j_index(&self) -> PyResult<usize> {
        Ok(match &self.s {
            StaticEvent::VDJ(x) => x.j_index,
            StaticEvent::VJ(x) => x.j_index,
        })
    }

    #[getter]
    fn get_j_start_seq(&self) -> PyResult<usize> {
        Ok(match &self.s {
            StaticEvent::VDJ(x) => x.j_start_seq,
            StaticEvent::VJ(x) => x.j_start_seq,
        })
    }

    #[getter]
    fn get_delj(&self) -> PyResult<usize> {
        Ok(match &self.s {
            StaticEvent::VDJ(x) => x.delj,
            StaticEvent::VJ(x) => x.delj,
        })
    }

    #[getter]
    fn get_d_index(&self) -> PyResult<usize> {
        match &self.s {
            StaticEvent::VDJ(x) => Ok(x.d_index),
            StaticEvent::VJ(_) => Err(anyhow!("No D index in a VJ model"))?,
        }
    }

    #[getter]
    fn get_d_start_seq(&self) -> PyResult<usize> {
        match &self.s {
            StaticEvent::VDJ(x) => Ok(x.d_start_seq),
            StaticEvent::VJ(_) => Err(anyhow!("No D sequence in a VJ model"))?,
        }
    }

    #[getter]
    fn get_deld3(&self) -> PyResult<usize> {
        match &self.s {
            StaticEvent::VDJ(x) => Ok(x.deld3),
            StaticEvent::VJ(_) => Err(anyhow!("No delD3 sequence in a VJ model"))?,
        }
    }

    #[getter]
    fn get_deld5(&self) -> PyResult<usize> {
        match &self.s {
            StaticEvent::VDJ(x) => Ok(x.deld5),
            StaticEvent::VJ(_) => Err(anyhow!("No delD5 sequence in a VJ model"))?,
        }
    }

    #[getter]
    fn get_insvd(&self) -> PyResult<Dna> {
        match &self.s {
            StaticEvent::VDJ(x) => Ok(x.insvd.clone()),
            StaticEvent::VJ(_) => Err(anyhow!("No VD insertions in a VJ model"))?,
        }
    }

    #[getter]
    fn get_insdj(&self) -> PyResult<Dna> {
        match &self.s {
            StaticEvent::VDJ(x) => Ok(x.insdj.clone()),
            StaticEvent::VJ(_) => Err(anyhow!("No DJ insertions in a VJ model"))?,
        }
    }

    #[getter]
    fn get_insvj(&self) -> PyResult<Dna> {
        match &self.s {
            StaticEvent::VJ(x) => Ok(x.insvj.clone()),
            StaticEvent::VDJ(_) => Err(anyhow!("No VJ insertions in a VDJ model"))?,
        }
    }

    #[getter]
    fn get_errors(&self) -> PyResult<Vec<(usize, char)>> {
        Ok(match &self.s {
            StaticEvent::VDJ(x) => x
                .errors
                .iter()
                .map(|(num, byte)| (*num, *byte as char))
                .collect(),
            StaticEvent::VJ(x) => x
                .errors
                .iter()
                .map(|(num, byte)| (*num, *byte as char))
                .collect(),
        })
    }
}

impl Default for StaticEvent {
    fn default() -> StaticEvent {
        StaticEvent::VDJ(VDJStaticEvent::default())
    }
}

impl StaticEvent {
    pub fn set_errors(&mut self, errors: Vec<(usize, u8)>) {
        match self {
            StaticEvent::VDJ(x) => x.errors = errors,
            StaticEvent::VJ(x) => x.errors = errors,
        }
    }

    pub fn to_sequence(&mut self, model: Model) -> Result<Dna> {
        match self {
            StaticEvent::VDJ(x) => match model {
                Model::VDJ(m) => Ok(x.to_sequence(&m)),
                Model::VJ(_) => Err(anyhow!("Wrong model type to recreate the sequence")),
            },
            StaticEvent::VJ(x) => match model {
                Model::VDJ(_) => Err(anyhow!("Wrong model type to recreate the sequence")),
                Model::VJ(m) => Ok(x.to_sequence(&m)),
            },
        }
    }

    pub fn extract_cdr3(&self, full_seq: &Dna, model: &Model) -> Result<Dna> {
        match (self, &model) {
            (StaticEvent::VDJ(ev), Model::VDJ(m)) => Ok(ev.extract_cdr3(full_seq, &m)),
            (StaticEvent::VJ(ev), Model::VJ(m)) => Ok(ev.extract_cdr3(full_seq, &m)),
            _ => Err(anyhow!("Wrong model for this event type")),
        }
    }
}
