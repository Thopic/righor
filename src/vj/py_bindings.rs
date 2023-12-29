//! Contains some of the python binding that would otherwise pollute the other files.

use crate::shared::GenerationResult;
use crate::vj::Model;
#[cfg(all(feature = "py_binds", feature = "py_o3"))]
use pyo3::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pyclass)]
pub struct Generator {
    pub model: Model,
    pub rng: SmallRng,
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

#[cfg_attr(all(feature = "py_binds", feature = "py_o3"), pymethods)]
impl Generator {
    pub fn generate(&mut self, functional: bool) -> GenerationResult {
        let (cdr3_nt, cdr3_aa, v_index, j_index) = self.model.generate(functional, &mut self.rng);
        let (full_sequence, v_name, j_name) = self
            .model
            .recreate_full_sequence(&cdr3_nt, v_index, j_index);
        GenerationResult {
            full_seq: full_sequence.to_string(),
            cdr3_nt: cdr3_nt.to_string(),
            cdr3_aa: cdr3_aa.map(|x| x.to_string()),
            v_gene: v_name,
            j_gene: j_name,
        }
    }
}

// Boiler-plate code for python bindings
#[cfg(all(feature = "py_binds", feature = "py_o3"))]
#[pymethods]
impl Generator {
    #[new]
    fn py_new(
        path_params: &str,
        path_marginals: &str,
        path_v_anchors: &str,
        path_j_anchors: &str,
        seed: Option<u64>,
    ) -> Generator {
        Generator::new(
            path_params,
            path_marginals,
            path_v_anchors,
            path_j_anchors,
            seed,
        )
    }
}
