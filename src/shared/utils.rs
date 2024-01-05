use crate::sequence::utils::{Dna, NUCLEOTIDES};
use anyhow::{anyhow, Result};
use ndarray::{s, Array, Array1, Array2, Array3, Axis, Dimension};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rand_distr::WeightedAliasIndex;

const EPSILON: f64 = 1e-10;

// Define some storage wrapper for the V/D/J genes

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug)]
pub struct Gene {
    pub name: String,
    pub seq: Dna,
    pub seq_with_pal: Option<Dna>, // Dna with the palindromic insertions (model dependant)
    pub functional: String,
    pub cdr3_pos: Option<usize>, // start (for V gene) or end (for J gene) of CDR3
}

impl Gene {
    pub fn create_palindromic_ends(&mut self, lenleft: usize, lenright: usize) {
        let palindromic_extension_left = self
            .seq
            .extract_subsequence(0, lenleft)
            .reverse_complement();
        let mut seqpal: Vec<u8> = palindromic_extension_left
            .seq
            .into_iter()
            .chain(self.seq.seq.clone())
            .collect();
        let palindromic_extension_right = self
            .seq
            .extract_subsequence(self.seq.len() - lenright, self.seq.len())
            .reverse_complement();
        seqpal.extend(palindromic_extension_right.seq);

        self.seq_with_pal = Some(Dna {
            seq: seqpal.clone(),
        });
    }
}

// Define the random distributions used in the rest of the code.

// Generate an integer with a given probability
#[derive(Clone, Debug)]
pub struct DiscreteDistribution {
    distribution: WeightedAliasIndex<f64>,
}

impl DiscreteDistribution {
    pub fn new(weights: Vec<f64>) -> Result<Self> {
        if !weights.iter().all(|&x| x >= 0.) {
            return Err(anyhow!(
                "Error when creating distribution: negative weights"
            ))?;
        }

        let distribution = match weights.iter().sum::<f64>().abs() < 1e-10 {
	    true => WeightedAliasIndex::new(vec![1.; weights.len()]) // when all the value are 0, all the values are equiprobable.
		.map_err(|e| anyhow!(format!("Error when creating distribution: {}", e)))?,
	    false => WeightedAliasIndex::new(weights)
		.map_err(|e| anyhow!(format!("Error when creating distribution: {}", e)))?
	};
        Ok(DiscreteDistribution { distribution })
    }

    pub fn generate<R: Rng>(&mut self, rng: &mut R) -> usize {
        self.distribution.sample(rng)
    }
}

impl Default for DiscreteDistribution {
    fn default() -> Self {
        DiscreteDistribution {
            distribution: WeightedAliasIndex::new(vec![1.]).unwrap(),
        }
    }
}

// Markov chain structure (for the insertion process)
#[derive(Default, Clone, Debug)]
pub struct MarkovDNA {
    initial_distribution: DiscreteDistribution, // first nucleotide, ACGT order
    transition_matrix: Vec<DiscreteDistribution>, // Markov matrix, ACGT order
}

impl MarkovDNA {
    pub fn new(transition_probs: Array2<f64>, initial_probs: Option<Vec<f64>>) -> Result<Self> {
        let mut transition_matrix = Vec::with_capacity(transition_probs.dim().0);
        for probs in transition_probs.axis_iter(Axis(0)) {
            transition_matrix.push(DiscreteDistribution::new(probs.to_vec())?);
        }
        let initial_distribution = match initial_probs {
            None => DiscreteDistribution::new(calc_steady_state_dist(&transition_probs)?)?,
            Some(dist) => DiscreteDistribution::new(dist)?,
        };
        Ok(MarkovDNA {
            initial_distribution,
            transition_matrix,
        })
    }

    pub fn generate<R: Rng>(&mut self, length: usize, rng: &mut R) -> Dna {
        let mut dna = Dna {
            seq: Vec::with_capacity(length),
        };
        if length == 0 {
            return dna;
        }

        let mut current_state = self.initial_distribution.generate(rng);
        dna.seq.push(NUCLEOTIDES[current_state]);

        for _ in 1..length {
            current_state = self.transition_matrix[current_state].generate(rng);
            dna.seq.push(NUCLEOTIDES[current_state]);
        }
        dna
    }
}

pub fn calc_steady_state_dist(transition_matrix: &Array2<f64>) -> Result<Vec<f64>> {
    // this should be profondly modified TODO
    // originally computed the eigenvalues. This is a pain though, because
    // it means I need to load blas, which takes forever to compile.
    // And this is not exactly an important part of the program.
    // so this means I'm going to do it stupidly

    // first normalize the transition matrix
    let mat = normalize_transition_matrix(transition_matrix)?;

    if mat.sum() == 0.0 {
        return Ok(vec![0.; mat.dim().0]);
    }

    let n = mat.nrows();
    let mut vec = Array1::from_elem(n, 1.0 / n as f64);
    for _ in 0..10000 {
        let vec_next = mat.dot(&vec);
        let norm = vec_next.sum();
        let vec_next = vec_next / norm;

        if (&vec_next - &vec).mapv(|a| a.abs()).sum() < EPSILON {
            return Ok(vec_next.to_vec());
        }
        vec = vec_next;
    }
    Err(anyhow!("No suitable eigenvector found"))?
}

pub fn add_errors<R: Rng>(dna: &mut Dna, error_rate: f64, rng: &mut R) {
    let effective_error_rate = error_rate * 4. / 3.;
    let uniform = Uniform::new(0.0, 1.0);
    let random_nucleotide = Uniform::new_inclusive(0, 3);

    for nucleotide in dna.seq.iter_mut() {
        if uniform.sample(rng) < effective_error_rate {
            *nucleotide = NUCLEOTIDES[random_nucleotide.sample(rng)];
        }
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
            // return a uniform distribution
            return Ok(Array2::zeros(self.dim()) / ((self.dim().0 * self.dim().1) as f64));
        }

        Ok(self / sum)
    }
}

impl Normalize2 for Array3<f64> {
    ///```
    /// use ndarray::{array, Array3};
    /// use ihor::shared::utils::Normalize2;
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
    if tm.iter().any(|&x| !x.is_finite()) {
        return Err(anyhow!("Array contains non-positive or non-finite values"));
    }
    let mut normalized = Array2::<f64>::zeros(tm.dim());
    for kk in 0..tm.dim().0 {
        let sum = tm.slice(s![kk, ..]).sum();
        if sum.abs() == 0.0f64 {
            for ii in 0..tm.dim().1 {
                normalized[[kk, ii]] = 0.; //1. / ((tm.dim().1) as f64);
            }
        } else {
            for ii in 0..tm.dim().1 {
                normalized[[kk, ii]] = tm[[kk, ii]] / sum;
            }
        }
    }
    Ok(normalized)
}

impl Normalize for Array2<f64> {
    /// Normalizes the elements of an array along the first axis.
    /// ```
    /// use ndarray::{array, Array2};
    /// use ihor::shared::utils::Normalize;
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

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug)]
pub struct InferenceParameters {
    pub min_log_likelihood: f64,
    pub evaluate: bool,
    pub nb_best_events: usize,
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl InferenceParameters {
    #[new]
    pub fn py_new(min_log_likelihood: f64) -> Self {
        Self::new(min_log_likelihood)
    }
}

impl InferenceParameters {
    pub fn new(min_likelihood: f64) -> Self {
        Self {
            min_log_likelihood: min_likelihood.log2(),
            evaluate: true,
            nb_best_events: 1,
        }
    }
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

/// Implement an array structure containing f64, indexed by min..max where min/max are i64
pub struct RangeArray1 {
    array: Vec<f64>,
    pub min: i64,
    pub max: i64, // over extremitie of the range (min + array.len())
}

impl RangeArray1 {
    pub fn new(values: &Vec<(i64, f64)>) -> RangeArray1 {
        if values.is_empty() {
            return RangeArray1 {
                min: 0,
                max: 0,
                array: Vec::new(),
            };
        }

        let min = values.iter().map(|x| x.0).min().unwrap();
        let max = values.iter().map(|x| x.0).max().unwrap() + 1;
        let mut array = vec![0.; (max - min) as usize];

        for (idx, value) in values {
            array[(idx - min) as usize] += value;
        }

        RangeArray1 { array, min, max }
    }

    pub fn get(&self, idx: i64) -> f64 {
        if idx < self.min || idx >= self.max {
            panic!(
                "index out of bounds: {} not in [{}, {}[",
                idx, self.min, self.max
            );
        }

        self.array[(idx - self.min) as usize]
    }

    pub fn dim(&self) -> (i64, i64) {
        (self.min, self.max)
    }

    pub fn zeros(range: (i64, i64)) -> RangeArray1 {
        RangeArray1 {
            min: range.0,
            max: range.1,
            array: vec![0.; (range.1 - range.0) as usize],
        }
    }

    pub fn get_mut(&mut self, idx: i64) -> &mut f64 {
        if idx < self.min || idx >= self.max {
            panic!(
                "index out of bounds: {} not in [{}, {}[",
                idx, self.min, self.max
            );
        }

        self.array.get_mut((idx - self.min) as usize).unwrap()
    }

    pub fn mut_map<F>(&mut self, mut f: F)
    where
        F: FnMut(f64) -> f64,
    {
        self.array.iter_mut().for_each(|x| *x = f(*x));
    }
}

/// Implement an array structure containing f64, indexed by min..max where min/max are i64
pub struct RangeArray3 {
    array: Vec<f64>,
    pub min: (i64, i64, i64),
    pub max: (i64, i64, i64),
    nb0: usize,
    nb1: usize,
}

impl RangeArray3 {
    pub fn new(values: &Vec<((i64, i64, i64), f64)>) -> RangeArray3 {
        if values.is_empty() {
            return RangeArray3 {
                min: (0, 0, 0),
                max: (0, 0, 0),
                nb0: 0,
                nb1: 0,
                array: Vec::new(),
            };
        }

        let min = (
            values.iter().map(|x| (x.0).0).min().unwrap(),
            values.iter().map(|x| (x.0).1).min().unwrap(),
            values.iter().map(|x| (x.0).2).min().unwrap(),
        );
        let max = (
            values.iter().map(|x| x.0 .0).max().unwrap() + 1,
            values.iter().map(|x| x.0 .1).max().unwrap() + 1,
            values.iter().map(|x| x.0 .2).max().unwrap() + 1,
        );
        let nb0 = (max.0 - min.0) as usize;
        let nb1 = (max.1 - min.1) as usize;

        let mut array = vec![0.; nb0 * nb1 * (max.2 - min.2) as usize];
        for ((i0, i1, i2), value) in values {
            array[(i0 - min.0) as usize
                + ((i1 - min.1) as usize) * nb0
                + ((i2 - min.2) as usize) * nb1 * nb0] += value;
        }
        RangeArray3 {
            array,
            min,
            max,
            nb0,
            nb1,
        }
    }

    pub fn get(&self, idx: (i64, i64, i64)) -> f64 {
        if idx.0 < self.min.0 || idx.0 >= self.max.0 {
            panic!(
                "index out of bounds: {} not in [{}, {}[",
                idx.0, self.min.0, self.max.0
            );
        }
        if idx.1 < self.min.1 || idx.1 >= self.max.1 {
            panic!(
                "index out of bounds: {} not in [{}, {}[",
                idx.1, self.min.1, self.max.1
            );
        }
        if idx.2 < self.min.2 || idx.2 >= self.max.2 {
            panic!(
                "index out of bounds: {} not in [{}, {}[",
                idx.2, self.min.2, self.max.2
            );
        }

        self.array[(idx.0 - self.min.0) as usize
            + ((idx.1 - self.min.1) as usize) * self.nb0
            + ((idx.2 - self.min.2) as usize) * self.nb1 * self.nb0]
    }

    pub fn get_mut(&mut self, idx: (i64, i64, i64)) -> &mut f64 {
        if idx.0 < self.min.0 || idx.0 >= self.max.0 {
            panic!(
                "index out of bounds: {} not in [{}, {}[",
                idx.0, self.min.0, self.max.0
            );
        }
        if idx.1 < self.min.1 || idx.1 >= self.max.1 {
            panic!(
                "index out of bounds: {} not in [{}, {}[",
                idx.1, self.min.1, self.max.1
            );
        }
        if idx.2 < self.min.2 || idx.2 >= self.max.2 {
            panic!(
                "index out of bounds: {} not in [{}, {}[",
                idx.2, self.min.2, self.max.2
            );
        }
        self.array
            .get_mut(
                (idx.0 - self.min.0) as usize
                    + ((idx.1 - self.min.1) as usize) * self.nb0
                    + ((idx.2 - self.min.2) as usize) * self.nb1 * self.nb0,
            )
            .unwrap()
    }

    pub fn dim(&self) -> ((i64, i64, i64), (i64, i64, i64)) {
        (self.min, self.max)
    }

    pub fn zeros(range: ((i64, i64, i64), (i64, i64, i64))) -> RangeArray3 {
        RangeArray3 {
            min: range.0,
            max: range.1,
            nb0: (range.1 .0 - range.0 .0) as usize,
            nb1: (range.1 .1 - range.0 .1) as usize,
            array: vec![
                0.;
                ((range.1 .0 - range.0 .0) * (range.1 .1 - range.0 .1) * (range.1 .2 - range.0 .2))
                    as usize
            ],
        }
    }
    pub fn mut_map<F>(&mut self, mut f: F)
    where
        F: FnMut(f64) -> f64,
    {
        self.array.iter_mut().for_each(|x| *x = f(*x));
    }
}

/// Implement an array structure containing f64, indexed by min..max where min/max are i64
pub struct RangeArray2 {
    array: Vec<f64>,
    pub min: (i64, i64),
    pub max: (i64, i64),
    nb0: usize,
}

impl RangeArray2 {
    pub fn new(values: &Vec<((i64, i64), f64)>) -> RangeArray2 {
        if values.is_empty() {
            return RangeArray2 {
                min: (0, 0),
                max: (0, 0),
                nb0: 0,
                array: Vec::new(),
            };
        }
        let min = (
            values.iter().map(|x| x.0 .0).min().unwrap(),
            values.iter().map(|x| x.0 .1).min().unwrap(),
        );
        let max = (
            values.iter().map(|x| x.0 .0).max().unwrap() + 1,
            values.iter().map(|x| x.0 .1).max().unwrap() + 1,
        );
        let nb0 = (max.0 - min.0) as usize;

        let mut array = vec![0.; nb0 * (max.1 - min.1) as usize];
        for ((i0, i1), value) in values {
            array[(i0 - min.0) as usize + (i1 - min.1) as usize * nb0] += value;
        }
        RangeArray2 {
            array,
            min,
            max,
            nb0,
        }
    }

    pub fn get(&self, idx: (i64, i64)) -> f64 {
        if idx.0 < self.min.0 || idx.0 >= self.max.0 {
            panic!(
                "index out of bounds: {} not in [{}, {}[",
                idx.0, self.min.0, self.max.0
            );
        }
        if idx.1 < self.min.1 || idx.1 >= self.max.1 {
            panic!(
                "index out of bounds: {} not in [{}, {}[",
                idx.1, self.min.1, self.max.1
            );
        }

        self.array[(idx.0 - self.min.0) as usize + (idx.1 - self.min.1) as usize * self.nb0]
    }

    pub fn get_mut(&mut self, idx: (i64, i64)) -> &mut f64 {
        if idx.0 < self.min.0 || idx.0 >= self.max.0 {
            panic!(
                "index out of bounds: {} not in [{}, {}[",
                idx.0, self.min.0, self.max.0
            );
        }
        if idx.1 < self.min.1 || idx.1 >= self.max.1 {
            panic!(
                "index out of bounds: {} not in [{}, {}[",
                idx.1, self.min.1, self.max.1
            );
        }

        self.array
            .get_mut((idx.0 - self.min.0) as usize + (idx.1 - self.min.1) as usize * self.nb0)
            .unwrap()
    }

    pub fn dim(&self) -> ((i64, i64), (i64, i64)) {
        (self.min, self.max)
    }

    pub fn zeros(range: ((i64, i64), (i64, i64))) -> RangeArray2 {
        RangeArray2 {
            min: range.0,
            max: range.1,
            nb0: (range.1 .0 - range.0 .0) as usize,
            array: vec![0.; ((range.1 .0 - range.0 .0) * (range.1 .1 - range.0 .1)) as usize],
        }
    }

    pub fn mut_map<F>(&mut self, mut f: F)
    where
        F: FnMut(f64) -> f64,
    {
        self.array.iter_mut().for_each(|x| *x = f(*x));
    }
}
