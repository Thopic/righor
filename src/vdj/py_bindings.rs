//! Contains some of the python binding that would otherwise pollute the other files.

use crate::sequence::AminoAcid;
use crate::vdj::{Model, StaticEvent};
use std::path::Path;

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

use rand::rngs::SmallRng;
use rand::SeedableRng;

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass)]
pub struct Generator {
    model: Model,
    rng: SmallRng,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub cdr3_nt: String,
    pub cdr3_aa: Option<String>,
    pub full_seq: String,
    pub v_gene: String,
    pub j_gene: String,
    pub recombination_event: StaticEvent,
}

impl Generator {
    pub fn new(model: Model, seed: Option<u64>) -> Generator {
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };

        Generator { model, rng }
    }
}

#[cfg(feature = "py_binds")]
#[pymethods]
impl Generator {
    #[new]
    pub fn py_new(
        path_params: &str,
        path_marginals: &str,
        path_v_anchors: &str,
        path_j_anchors: &str,
        seed: Option<u64>,
    ) -> Generator {
        Generator::new(
            Model::load_from_files(
                Path::new(path_params),
                Path::new(path_marginals),
                Path::new(path_v_anchors),
                Path::new(path_j_anchors),
            )
            .unwrap(),
            seed,
        )
    }
}

#[cfg_attr(features = "py_bind", pymethods)]
impl Generator {
    pub fn generate(&mut self, functional: bool) -> GenerationResult {
        let (cdr3_nt, cdr3_aa, full_sequence, event, vname, jname) =
            self.model.generate(functional, &mut self.rng);
        GenerationResult {
            full_seq: full_sequence.to_string(),
            cdr3_nt: cdr3_nt.to_string(),
            cdr3_aa: cdr3_aa.map(|x: AminoAcid| x.to_string()),
            v_gene: vname,
            j_gene: jname,
            recombination_event: event,
        }
    }
}
