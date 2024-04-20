//! The structs used for specifying the parameters of the model
use bio::alignment::{pairwise, Alignment};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;


#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
pub struct AlignmentParameters {
    // Structure containing all the parameters for the alignment
    // of the V and J genes
    pub min_score_v: i32,
    pub min_score_j: i32,
    pub max_error_d: usize,
    pub left_v_cutoff: usize,
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Clone, Debug)]
pub struct InferenceParameters {
    // The evaluation/inference algorithm will cut branches
    // with likelihood < min_likelihood
    pub min_likelihood: f64,
    // The evaluation/inference algorithm will cut branches with
    // likelihood < (best current likelihood * min_ratio_likelihood)
    pub min_ratio_likelihood: f64,
    // If true, run the inference (so update the features)
    pub infer: bool,
    // If true store the highest likelihood event
    pub store_best_event: bool,
    // If true and "store_best_event" is true, compute the pgen of the sequence
    // (pgen is computed by default if the model error rate is 0)
    pub compute_pgen: bool,

    // If true (default) infer the insertion distribution
    pub infer_insertions: bool,
    // If true (default) infer the deletion distribution & gene usage
    pub infer_genes: bool,
}


impl Default for AlignmentParameters {
    fn default() -> AlignmentParameters {
        AlignmentParameters {
            min_score_v: -20,
            min_score_j: 0,
            max_error_d: 100,
            left_v_cutoff: 50,
        }
    }
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl AlignmentParameters {
    #[new]
    pub fn py_new() -> Self {
        AlignmentParameters::default()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "AlignmentParameters(min_score_v={}, min_score_j={}, max_error_d={}. left_v_cutoff={})",
            self.min_score_v, self.min_score_j, self.max_error_d, self.left_v_cutoff
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        // This is what will be shown when you use print() in Python
        self.__repr__()
    }
}

impl AlignmentParameters {
    pub fn new(
        min_score_v: i32,
        min_score_j: i32,
        max_error_d: usize,
        left_v_cutoff: usize,
    ) -> Self {
        Self {
            min_score_v,
            min_score_j,
            max_error_d,
            left_v_cutoff, // shorten the V gene for alignment (improve speed)
        }
    }

    pub fn get_scoring(&self) -> pairwise::Scoring<Box<dyn Fn(u8, u8) -> i32>> {
        pairwise::Scoring {
            gap_open: -100,
            gap_extend: -20,
            // TODO: deal better with possible IUPAC codes
            match_fn: Box::new(|a: u8, b: u8| {
                if a == b {
                    6i32
                } else if (a == b'N') | (b == b'N') {
                    0i32
                } else {
                    -3i32
                }
            }),
            match_scores: None,
            xclip_prefix: 0,
            xclip_suffix: pairwise::MIN_SCORE,
            yclip_prefix: pairwise::MIN_SCORE,
            yclip_suffix: 0,
        }
    }

    pub fn get_scoring_local(&self) -> pairwise::Scoring<Box<dyn Fn(u8, u8) -> i32>> {
        pairwise::Scoring {
            gap_open: -50,
            gap_extend: -10,
            // TODO: deal better with possible IUPAC codes
            match_fn: Box::new(|a: u8, b: u8| {
                if a == b {
                    6i32
                } else if (a == b'N') | (b == b'N') {
                    0i32
                } else {
                    -3i32
                }
            }),
            match_scores: None,
            xclip_prefix: 0,
            xclip_suffix: pairwise::MIN_SCORE, // still need V to go to the end
            yclip_prefix: 0,
            yclip_suffix: 0,
        }
    }

    pub fn valid_v_alignment(&self, al: &Alignment) -> bool {
        al.xend - al.xstart == al.yend - al.ystart
    }

    pub fn valid_j_alignment(&self, al: &Alignment) -> bool {
        // right now: no insert
        al.score > self.min_score_j && al.xend - al.xstart == al.yend - al.ystart
    }
}


#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl InferenceParameters {
    #[new]
    pub fn py_new() -> Self {
        InferenceParameters::default()
    }
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("InferenceParameters(min_likelihood={}, min_ratio_likelihood={}, infer={}, store_best_event={}, compute_pgen={})", self.min_likelihood, self.min_ratio_likelihood, self.infer, self.store_best_event, self.compute_pgen))
    }
    fn __str__(&self) -> PyResult<String> {
        // This is what will be shown when you use print() in Python
        self.__repr__()
    }
}

impl Default for InferenceParameters {
    fn default() -> InferenceParameters {
        InferenceParameters {
            min_likelihood: (-400.0f64).exp2(),
            min_ratio_likelihood: (-100.0f64).exp2(),
            infer: true,
            store_best_event: true,
            compute_pgen: true,
	    infer_insertions: true,
	    infer_genes: true,
        }
    }
}

impl InferenceParameters {
    pub fn new(min_likelihood: f64) -> Self {
        Self {
            min_likelihood,
            ..Default::default()
        }
    }
}
