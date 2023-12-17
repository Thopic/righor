#![allow(unused_imports)] //TODO REMOVE
#![allow(dead_code)]

mod model;
mod parser;
mod sequence;
mod utils;

mod utils_sequences;
use model::{ModelVDJ, ModelVJ};
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::path::Path;
use utils_sequences::AminoAcid;

#[pyclass]
#[derive(Debug, Clone)]
pub struct GenerationResult {
    #[pyo3(get, set)]
    pub cdr3_nt: String,
    #[pyo3(get, set)]
    pub cdr3_aa: Option<String>,
    #[pyo3(get, set)]
    pub full_seq: String,
    #[pyo3(get, set)]
    pub v_gene: String,
    #[pyo3(get, set)]
    pub j_gene: String,
}

#[pyclass]
pub struct GeneratorVDJ {
    model: ModelVDJ,
    rng: SmallRng,
}

#[pyclass]
pub struct GeneratorVJ {
    model: ModelVJ,
    rng: SmallRng,
}

#[pymethods]
impl GeneratorVDJ {
    #[new]
    pub fn new(
        path_params: &str,
        path_marginals: &str,
        path_v_anchors: &str,
        path_j_anchors: &str,
        seed: Option<u64>,
    ) -> GeneratorVDJ {
        let model = ModelVDJ::load_model(
            &Path::new(path_params),
            &Path::new(path_marginals),
            &Path::new(path_v_anchors),
            &Path::new(path_j_anchors),
        )
        .unwrap();
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };

        GeneratorVDJ {
            model: model,
            rng: rng,
        }
    }

    pub fn generate(&mut self, functional: bool) -> GenerationResult {
        let (cdr3_nt, cdr3_aa, v_index, j_index) = self.model.generate(functional, &mut self.rng);
        let (full_sequence, v_name, j_name) =
            self.model
                .recreate_full_sequence(cdr3_nt.clone(), v_index, j_index);
        return GenerationResult {
            full_seq: full_sequence.to_string(),
            cdr3_nt: cdr3_nt.to_string(),
            cdr3_aa: cdr3_aa.map(|x: AminoAcid| x.to_string()),
            v_gene: v_name,
            j_gene: j_name,
        };
    }
}

#[pymethods]
impl GeneratorVJ {
    #[new]
    fn new(
        path_params: &str,
        path_marginals: &str,
        path_v_anchors: &str,
        path_j_anchors: &str,
        seed: Option<u64>,
    ) -> GeneratorVJ {
        let model = ModelVJ::load_model(
            &Path::new(path_params),
            &Path::new(path_marginals),
            &Path::new(path_v_anchors),
            &Path::new(path_j_anchors),
        )
        .unwrap();
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };
        GeneratorVJ {
            model: model,
            rng: rng,
        }
    }

    fn generate(&mut self, functional: bool) -> GenerationResult {
        let (cdr3_nt, cdr3_aa, v_index, j_index) = self.model.generate(functional, &mut self.rng);
        let (full_sequence, v_name, j_name) =
            self.model
                .recreate_full_sequence(cdr3_nt.clone(), v_index, j_index);
        return GenerationResult {
            full_seq: full_sequence.to_string(),
            cdr3_nt: cdr3_nt.to_string(),
            cdr3_aa: cdr3_aa.map(|x| x.to_string()),
            v_gene: v_name,
            j_gene: j_name,
        };
    }
}

#[pymodule]
fn ihor(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<GeneratorVDJ>()?;
    m.add_class::<GeneratorVJ>()?;
    m.add_class::<GenerationResult>()?;
    Ok(())
}
