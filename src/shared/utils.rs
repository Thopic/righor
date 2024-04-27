use anyhow::{anyhow, Result};

use ndarray::{s, Array, Array1, Array2, Array3, Dimension};

use rayon;
use serde::{Deserialize, Serialize};

pub fn fix_number_threads(nb_threads: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(nb_threads)
        .build_global()
        .unwrap();
}

pub fn count_differences<T: PartialEq>(vec1: &[T], vec2: &[T]) -> usize {
    vec1.iter()
        .zip(vec2.iter())
        .filter(|&(a, b)| a != b)
        .count()
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordModel {
    pub species: Vec<String>,
    pub chain: Vec<String>,
    pub id: String,
    pub filename_params: String,
    pub filename_marginals: String,
    pub filename_v_gene_cdr3_anchors: String,
    pub filename_j_gene_cdr3_anchors: String,
    pub description: String,
}

/// Normalize the distribution on the last axis
pub trait NormalizeLast {
    fn normalize_last(&self) -> Result<Self>
    where
        Self: Sized;
}

/// Normalize the distribution on the two last axis
pub trait NormalizeLast2 {
    fn normalize_last_2(&self) -> Result<Self>
    where
        Self: Sized;
}

/// Normalize the distribution on the first three axis
pub trait Normalize3 {
    fn normalize_distribution_3(&self) -> Result<Self>
    where
        Self: Sized;
}

impl Normalize3 for Array3<f64> {
    fn normalize_distribution_3(&self) -> Result<Self> {
        if self.iter().any(|&x| x < 0.0) {
            // negative values mean something wrong happened
            return Err(anyhow!("Array contains non-positive values"));
        }

        let sum = self.sum();
        if sum.abs() == 0.0f64 {
            // return a uniform distribution
            return Ok(Array3::zeros(self.dim()));
        }

        Ok(self / sum)
    }
}

/// Normalize the distribution on the two first axis
pub trait Normalize2 {
    fn normalize_distribution_double(&self) -> Result<Self>
    where
        Self: Sized;
}

impl Normalize2 for Array2<f64> {
    fn normalize_distribution_double(&self) -> Result<Self> {
        if self.iter().any(|&x| x < 0.0) {
            // negative values mean something wrong happened
            return Err(anyhow!("Array contains non-positive values"));
        }

        let sum = self.sum();
        if sum.abs() == 0.0f64 {
            // return a zero distribution
            return Ok(Array2::zeros(self.dim()));
        }

        Ok(self / sum)
    }
}

impl Normalize2 for Array3<f64> {
    ///```
    /// use ndarray::{array, Array3};
    /// use righor::shared::utils::Normalize2;
    /// let a: Array3<f64> = array![[[1., 2., 3.], [1., 2., 3.], [3., 4., 5.]]];
    /// let b = a.normalize_distribution_double().unwrap();
    /// println!("{:?}", b);
    /// let truth =  array![[        [0.2, 0.25, 0.27272727],        [0.2, 0.25, 0.27272727],        [0.6, 0.5, 0.4545454]    ]];
    /// assert!( ((b.clone() - truth.clone())*(b-truth)).sum()< 1e-8);
    /// let a2: Array3<f64> = array![[[0., 0.], [2., 0.], [0., 0.], [0., 0.]]];
    /// let b2 = a2.normalize_distribution_double().unwrap();
    /// let truth2 = array![[[0., 0.], [1., 0.], [0., 0.], [0., 0.]]];
    /// println!("{:?}", b2);
    /// assert!( ((b2.clone() - truth2.clone())*(b2-truth2)).sum()< 1e-8);
    fn normalize_distribution_double(&self) -> Result<Self> {
        let mut normalized = Array3::<f64>::zeros(self.dim());
        for ii in 0..self.dim().2 {
            let sum = self.slice(s![.., .., ii]).sum();

            if sum.abs() == 0.0f64 {
                for jj in 0..self.dim().0 {
                    for kk in 0..self.dim().1 {
                        normalized[[jj, kk, ii]] = 0.;
                    }
                }
            } else {
                for jj in 0..self.dim().0 {
                    for kk in 0..self.dim().1 {
                        normalized[[jj, kk, ii]] = self[[jj, kk, ii]] / sum;
                    }
                }
            }
        }
        Ok(normalized)
    }
}

pub trait Normalize {
    fn normalize_distribution(&self) -> Result<Self>
    where
        Self: Sized;
}

impl Normalize for Array1<f64> {
    fn normalize_distribution(&self) -> Result<Self> {
        if self.iter().any(|&x| x < 0.0) {
            // negative values mean something wrong happened
            return Err(anyhow!("Array contains non-positive values"));
        }

        let sum = self.sum();
        if sum.abs() == 0.0f64 {
            // return a uniform distribution
            return Ok(Array1::zeros(self.dim()) / self.dim() as f64);
        }

        Ok(self / sum)
    }
}

/// Normalize the elements of the array along the second axis
/// equivalent of a/a.sum(axis=1)[:, np.newaxis] in numpy
pub fn normalize_transition_matrix(tm: &Array2<f64>) -> Result<Array2<f64>> {
    tm.normalize_last()
}

/// Normalize the elements of an array along the last axis
impl NormalizeLast for Array2<f64> {
    fn normalize_last(&self) -> Result<Self> {
        if self.iter().any(|&x| !x.is_finite()) {
            return Err(anyhow!("Array contains non-positive or non-finite values"));
        }
        let mut normalized = Array2::<f64>::zeros(self.dim());
        for ii in 0..self.dim().0 {
            let sum = self.slice(s![ii, ..]).sum();
            if sum.abs() == 0.0f64 {
                for kk in 0..self.dim().1 {
                    normalized[[ii, kk]] = 0.;
                }
            } else {
                for kk in 0..self.dim().1 {
                    normalized[[ii, kk]] = self[[ii, kk]] / sum;
                }
            }
        }
        Ok(normalized)
    }
}

/// Normalize the elements of an array along the last axis
impl NormalizeLast for Array3<f64> {
    fn normalize_last(&self) -> Result<Self> {
        if self.iter().any(|&x| !x.is_finite()) {
            return Err(anyhow!("Array contains non-positive or non-finite values"));
        }
        let mut normalized = Array3::<f64>::zeros(self.dim());
        for ii in 0..self.dim().0 {
            for jj in 0..self.dim().1 {
                let sum = self.slice(s![ii, jj, ..]).sum();
                if sum.abs() == 0.0f64 {
                    for kk in 0..self.dim().2 {
                        normalized[[ii, jj, kk]] = 0.;
                    }
                } else {
                    for kk in 0..self.dim().2 {
                        normalized[[ii, jj, kk]] = self[[ii, jj, kk]] / sum;
                    }
                }
            }
        }
        Ok(normalized)
    }
}

/// Normalize the elements of an array along the two last axis
impl NormalizeLast2 for Array3<f64> {
    fn normalize_last_2(&self) -> Result<Self> {
        let mut normalized = Array3::<f64>::zeros(self.dim());
        for ii in 0..self.dim().0 {
            let sum = self.slice(s![ii, .., ..]).sum();

            if sum.abs() == 0.0f64 {
                for jj in 0..self.dim().1 {
                    for kk in 0..self.dim().2 {
                        normalized[[ii, jj, kk]] = 0.;
                    }
                }
            } else {
                for jj in 0..self.dim().1 {
                    for kk in 0..self.dim().2 {
                        normalized[[ii, jj, kk]] = self[[ii, jj, kk]] / sum;
                    }
                }
            }
        }
        Ok(normalized)
    }
}

impl Normalize for Array2<f64> {
    /// Normalizes the elements of an array along the first axis.
    /// ```
    /// use ndarray::{array, Array2};
    /// use righor::shared::utils::Normalize;
    /// let a : Array2<f64> = array![[0.0, 2.0, 3.0], [2.0, 3.0, 3.0]];
    /// let result = a.normalize_distribution().unwrap();
    /// assert!(result == array![[0. , 0.4, 0.5],[1. , 0.6, 0.5]])
    /// ```
    fn normalize_distribution(&self) -> Result<Self> {
        if self.iter().any(|&x| !x.is_finite()) {
            return Err(anyhow!("Array contains non-positive or non-finite values"));
        }
        let mut normalized = Array2::<f64>::zeros(self.dim());
        for ii in 0..self.dim().1 {
            let sum = self.slice(s![.., ii]).sum();
            if sum.abs() == 0.0f64 {
                for kk in 0..self.dim().0 {
                    normalized[[kk, ii]] = 0.;
                }
            } else {
                for kk in 0..self.dim().0 {
                    normalized[[kk, ii]] = self[[kk, ii]] / sum;
                }
            }
        }
        Ok(normalized)
    }
}

impl Normalize for Array3<f64> {
    /// Normalizes the elements of an array along the first axis.
    fn normalize_distribution(&self) -> Result<Self> {
        if self.iter().any(|&x| !x.is_finite()) {
            return Err(anyhow!("Array contains non-positive or non-finite values"));
        }
        let mut normalized = Array3::<f64>::zeros(self.dim());
        for ii in 0..self.dim().1 {
            for jj in 0..self.dim().2 {
                let sum = self.slice(s![.., ii, jj]).sum();
                if sum.abs() == 0.0f64 {
                    for kk in 0..self.dim().0 {
                        normalized[[kk, ii, jj]] = 0.;
                    }
                } else {
                    for kk in 0..self.dim().0 {
                        normalized[[kk, ii, jj]] = self[[kk, ii, jj]] / sum;
                    }
                }
            }
        }
        Ok(normalized)
    }
}

pub fn sorted_and_complete(arr: Vec<i64>) -> bool {
    // check that the array is sorted and equal to
    // arr[0]..arr.last()
    if arr.is_empty() {
        return true;
    }
    let mut b = arr[0];
    for a in &arr[1..] {
        if *a != b + 1 {
            return false;
        }
        b = *a;
    }
    true
}

pub fn sorted_and_complete_0start(arr: Vec<i64>) -> bool {
    // check that the array is sorted and equal to
    // 0..arr.last()
    if arr.is_empty() {
        return true;
    }
    for (ii, a) in arr.iter().enumerate() {
        if *a != (ii as i64) {
            return false;
        }
    }
    true
}

/// Return a vector with a tuple (f64, T) inserted, so that the vector stays sorted
/// along the first element of the pair
/// # Arguments
/// * `v` – The original *decreasing* vector (which is going to be cloned and modified)
/// * `elem` – The element to be inserted
/// # Returns the vector v with elem inserted such that v.map(|x| x.0) is decreasing.
pub fn insert_in_order<T>(v: Vec<(f64, T)>, elem: (f64, T)) -> Vec<(f64, T)>
where
    T: Clone,
{
    let pos = v.binary_search_by(|(f, _)| {
        (-f).partial_cmp(&(-elem.0))
            .unwrap_or(std::cmp::Ordering::Less)
    });
    let index = match pos {
        Ok(i) | Err(i) => i,
    };
    let mut vcloned: Vec<(f64, T)> = v.to_vec();
    vcloned.insert(index, elem);
    vcloned
}

pub fn max_of_array<D>(arr: &Array<f64, D>) -> f64
where
    D: Dimension,
{
    if arr.is_empty() {
        return f64::NEG_INFINITY;
    }
    arr.into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .cloned()
        .unwrap()
}

pub fn max_f64(a: f64, b: f64) -> f64 {
    match (a, b) {
        // If either is NaN, return NaN
        (a, b) if a.is_nan() || b.is_nan() => f64::NAN,

        // If either is positive infinity, return positive infinity
        (a, b)
            if a.is_infinite() && a.is_sign_positive()
                || b.is_infinite() && b.is_sign_positive() =>
        {
            f64::INFINITY
        }

        // Normal max calculation otherwise
        _ => a.max(b),
    }
}

pub fn difference_as_i64(a: usize, b: usize) -> i64 {
    if a >= b {
        // don't check for overflow, trust the system
        // assert!(a - b <= i64::MAX as usize, "Overflow occurred");
        (a - b) as i64
    } else {
        // don't check for underflow either
        // assert!(b - a <= i64::MAX as usize, "Underflow occurred");
        -((b - a) as i64)
    }
}
