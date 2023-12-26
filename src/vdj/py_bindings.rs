//! Contains some of the python binding that would otherwise pollute the other files.

use crate::sequence::AminoAcid;
use crate::shared::GenerationResult;
use crate::vdj::Model;

use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::path::Path;

#[pyclass]
pub struct Generator {
    model: Model,
    rng: SmallRng,
}

#[pymethods]
impl Generator {
    #[new]
    pub fn new(
        path_params: &str,
        path_marginals: &str,
        path_v_anchors: &str,
        path_j_anchors: &str,
        seed: Option<u64>,
    ) -> Generator {
        let model = Model::load_model(
            Path::new(path_params),
            Path::new(path_marginals),
            Path::new(path_v_anchors),
            Path::new(path_j_anchors),
        )
        .unwrap();
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };

        Generator { model, rng }
    }

    pub fn generate(&mut self, functional: bool) -> GenerationResult {
        let (cdr3_nt, cdr3_aa, v_index, j_index) = self.model.generate(functional, &mut self.rng);
        let (full_sequence, v_name, j_name) = self
            .model
            .recreate_full_sequence(&cdr3_nt, v_index, j_index);
        GenerationResult {
            full_seq: full_sequence.to_string(),
            cdr3_nt: cdr3_nt.to_string(),
            cdr3_aa: cdr3_aa.map(|x: AminoAcid| x.to_string()),
            v_gene: v_name,
            j_gene: j_name,
        }
    }
}
