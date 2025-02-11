//! Define the "Likelihood" class that contain the likelihood of a DNA segment
//! as well as its container (`Likelihood1dContainer`, etc...)
//! When the DNA segment is only known through its amino-acid sequence, we don't
//! completely know the likelihood, as it can depend on the nucleotide that
//! are before, or after, the sequence.
//! For example, the amino-acid sequence SS between position [2; 6[ will have
//! different likelihoods depending on the value of the first two nucleotides
//!  AG[C, T] or TCN.
//! This is mostly a problem for the insertions (because the V/D/J genes are
//! fully known), but I treat everything with the same formalism.

use crate::shared::data_structures::{RangeArray1, RangeArray2};
use nalgebra;
pub type Vector16 = nalgebra::SVector<f64, 16>;
pub type Vector4 = nalgebra::SVector<f64, 4>;
pub type Matrix16 = nalgebra::SMatrix<f64, 16, 16>;
pub type Matrix4x16 = nalgebra::SMatrix<f64, 4, 16>;
pub type Matrix16x4 = nalgebra::SMatrix<f64, 16, 4>;
pub type Matrix4 = nalgebra::SMatrix<f64, 4, 4>;

use crate::shared::alignment::DAlignment;
use crate::shared::alignment::VJAlignment;
use crate::shared::sequence::DnaLike;
use crate::shared::sequence::SequenceType;
use anyhow::{anyhow, Result};
use core::ops;
use itertools::Either;
//use nohash_hasher::NoHashHasher;

use foldhash::{HashMap, HashMapExt};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;

#[cfg_attr(
    all(feature = "py_binds", feature = "pyo3"),
    pyclass(get_all, set_all, eq, eq_int)
)]
#[derive(Clone, Debug, Copy, Default, PartialEq)]
pub enum LikelihoodType {
    #[default]
    Scalar,
    Vector,
    Matrix,
}

#[derive(Clone, Debug)]
pub enum Likelihood {
    Scalar(f64),
    Vector(Box<Vector16>),
    Matrix(Box<Matrix16>),
}

// Implement Add for Likelihood + Likelihood
impl ops::Add for Likelihood {
    type Output = Likelihood;

    fn add(self, rhs: Likelihood) -> Likelihood {
        match (self, rhs) {
            (Likelihood::Scalar(lhs), Likelihood::Scalar(rhs)) => Likelihood::Scalar(lhs + rhs),
            (Likelihood::Vector(lhs), Likelihood::Vector(rhs)) => {
                Likelihood::Vector(Box::new(*lhs + *rhs))
            }
            (Likelihood::Matrix(lhs), Likelihood::Matrix(rhs)) => {
                Likelihood::Matrix(Box::new(*lhs + *rhs))
            }
            _ => panic!("Incompatible types for addition"),
        }
    }
}

// Implement AddAssign for Likelihood += Likelihood
impl ops::AddAssign for Likelihood {
    fn add_assign(&mut self, rhs: Likelihood) {
        match (self, rhs) {
            (Likelihood::Scalar(ref mut lhs), Likelihood::Scalar(rhs)) => *lhs += rhs,
            (Likelihood::Vector(ref mut lhs), Likelihood::Vector(rhs)) => **lhs += *rhs,
            (Likelihood::Matrix(ref mut lhs), Likelihood::Matrix(rhs)) => **lhs += *rhs,
            _ => panic!("Incompatible types for addition assignment"),
        }
    }
}

impl ops::Mul<Likelihood> for Likelihood {
    type Output = Likelihood;
    fn mul(self, rhs: Likelihood) -> Likelihood {
        match (self, rhs) {
            (Likelihood::Scalar(u), Likelihood::Scalar(v)) => Likelihood::Scalar(u * v),
            (Likelihood::Vector(u), Likelihood::Vector(v)) => {
                Likelihood::Scalar(((*u).transpose() * *v)[(0, 0)])
            }
            (Likelihood::Matrix(u), Likelihood::Matrix(v)) => Likelihood::Matrix(Box::new(*u * *v)),
            (Likelihood::Vector(u), Likelihood::Matrix(v)) => {
                Likelihood::Vector(Box::new(((*u).transpose() * *v).transpose()))
            }
            (Likelihood::Matrix(u), Likelihood::Vector(v)) => Likelihood::Vector(Box::new(*u * *v)),
            (_, _) => panic!("Incompatible types for vector multiplications"),
        }
    }
}

impl ops::Mul<f64> for Likelihood {
    type Output = Likelihood;
    fn mul(self, rhs: f64) -> Likelihood {
        match (self, rhs) {
            (Likelihood::Scalar(u), x) => Likelihood::Scalar(u * x),
            (Likelihood::Vector(u), x) => Likelihood::Vector(Box::new(*u * x)),
            (Likelihood::Matrix(u), x) => Likelihood::Matrix(Box::new(*u * x)),
        }
    }
}

// Implement Mul<f64> for the reverse case (f64 * Likelihood)
impl ops::Mul<Likelihood> for f64 {
    type Output = Likelihood;
    fn mul(self, rhs: Likelihood) -> Likelihood {
        rhs * self // Reuse the implementation
    }
}

impl Likelihood {
    // /// Return a matrix set to 1. for positions that match
    // /// the two nucleotides before / the two last nucleotides of the D gene.
    // /// Also deal with the situation where d is too short.
    // pub fn from_insertions(insvd: &DnaLike) -> Likelihood {
    //     let mut m = Matrix16::zeros();
    //     for (i1, i2) in insvd.valid_extremities() {
    //         m[(i1, i2)] = 1.;
    //     }
    //     Likelihood::Matrix(m)
    // }

    pub fn zero(dna: &DnaLike) -> Likelihood {
        if dna.is_protein() {
            Likelihood::Matrix(Box::new(Matrix16::zeros()))
        } else {
            Likelihood::Scalar(0.)
        }
    }

    pub fn identity(dna: &DnaLike) -> Likelihood {
        if dna.is_protein() {
            Likelihood::Matrix(Box::new(Matrix16::identity()))
        } else {
            Likelihood::Scalar(1.)
        }
    }

    /// Return a matrix set to 1. for positions that match
    /// the two nucleotides before / the two last nucleotides of the D gene.
    /// Also deal with the situation where d is too short.
    pub fn from_d_sides(d: &DAlignment, deld5: usize, deld3: usize) -> Likelihood {
        let mut m = Matrix16::zeros();
        for (idx1, idx2) in d.valid_extremities(deld5, deld3) {
            m[(idx1, idx2)] = 1.;
        }
        Likelihood::Matrix(Box::new(m))
    }

    /// Return a vector set to 1. for positions that match
    /// the two nucleotides before the J gene
    pub fn from_j_side(j: &VJAlignment, del: usize) -> Likelihood {
        let mut vec = Vector16::zeros();
        for idx in j.valid_extended_j(del) {
            vec[idx] = 1.;
        }
        Likelihood::Vector(Box::new(vec))
    }

    /// Return a vector set to 1 for position that match
    /// the last two nucleotides of the V gene.
    /// Only one coefficient of the vector is non-zero
    pub fn from_v_side(v: &VJAlignment, del: usize) -> Likelihood {
        let mut vec = Vector16::zeros();
        let end_v = v.gene_sequence.len() as i64 - del as i64;
        for idx in v
            .gene_sequence
            .extract_padded_subsequence(end_v - 2, end_v)
            .to_matrix_idx()
        {
            vec[idx] = 1.;
        }
        Likelihood::Vector(Box::new(vec))
    }

    pub fn max(&self) -> f64 {
        match self {
            Likelihood::Scalar(x) => *x,
            Likelihood::Vector(x) => x.sum(),
            Likelihood::Matrix(x) => x.sum(),
        }
    }
    pub fn is_zero(&self) -> bool {
        self.max() == 0.
    }

    pub fn keep(&self) -> bool {
        match self {
            Likelihood::Scalar(x) => *x > 0.,
            Likelihood::Vector(_) | Likelihood::Matrix(_) => true,
        }
    }

    pub fn to_scalar(&self) -> Result<f64> {
        match self {
            Likelihood::Scalar(x) => Ok(*x),
            _ => Err(anyhow!("This likelihood is not a scalar")),
        }
    }
    pub fn to_matrix(&self) -> Result<Matrix16> {
        match self {
            Likelihood::Matrix(x) => Ok(**x),
            _ => Err(anyhow!("This likelihood is not a matrix")),
        }
    }

    pub fn to_vector(&self) -> Result<Vector16> {
        match self {
            Likelihood::Vector(x) => Ok(**x),
            _ => Err(anyhow!("This likelihood is not a matrix")),
        }
    }

    pub fn zero_from_type(lt: LikelihoodType) -> Likelihood {
        match lt {
            LikelihoodType::Scalar => Likelihood::Scalar(0.),
            LikelihoodType::Vector => Likelihood::Vector(Box::new(Vector16::zeros())),
            LikelihoodType::Matrix => Likelihood::Matrix(Box::new(Matrix16::zeros())),
        }
    }

    /// Element-wise division
    pub fn divide(&self, rhs: Likelihood) -> Likelihood {
        match (self, rhs) {
            (Likelihood::Scalar(lhs), Likelihood::Scalar(rhs)) => Likelihood::Scalar(lhs / rhs),
            (Likelihood::Vector(lhs), Likelihood::Vector(rhs)) => {
                Likelihood::Vector(Box::new(lhs.component_div(&rhs)))
            }
            (Likelihood::Matrix(lhs), Likelihood::Matrix(rhs)) => {
                Likelihood::Matrix(Box::new(lhs.component_div(&rhs)))
            }
            _ => panic!("Incompatible types for addition"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Likelihood1DContainer {
    Scalar(RangeArray1), // for the normal nucleotides sequences
    Matrix(
        HashMap<
            i64,
            Vector16, // , nohash_hasher::BuildNoHashHasher<i64>
        >,
    ), // for the amino-acids
}

impl Likelihood1DContainer {
    pub fn dim(&self) -> (i64, i64) {
        (self.min(), self.max())
    }

    /// Get the value at position `pos`
    pub fn get(&self, pos: i64) -> Likelihood {
        match &self {
            Likelihood1DContainer::Scalar(x) => Likelihood::Scalar(x.get(pos)),
            Likelihood1DContainer::Matrix(x) => {
                //debug_assert!(pos >= self.min() && pos <= self.max());
                if !x.contains_key(&pos) {
                    Likelihood::Vector(Box::new(Vector16::zeros()))
                } else {
                    Likelihood::Vector(Box::new(*x.get(&pos).unwrap()))
                }
            }
        }
    }
    /// Add `likelihood` to the value at `pos`
    pub fn add_to(&mut self, pos: i64, likelihood: Likelihood) {
        match (self, likelihood) {
            (Likelihood1DContainer::Scalar(x), Likelihood::Scalar(ll)) => {
                *x.get_mut(pos) += ll;
            }
            (Likelihood1DContainer::Matrix(x), Likelihood::Vector(ll)) => {
                *x.entry(pos).or_insert(Vector16::zeros()) += *ll;
            }
            _ => unimplemented!("Something wrong with add_to 1Dcontainer"),
        }
    }

    /// Return a container with only 0 values
    pub fn zeros(start: i64, end: i64, dt: SequenceType) -> Likelihood1DContainer {
        match dt {
            SequenceType::Dna => Likelihood1DContainer::Scalar(RangeArray1::zeros((start, end))),
            SequenceType::Protein => {
                Likelihood1DContainer::Matrix(HashMap::// with_hasher(BuildHasherDefault::<
                //     NoHashHasher<i64>,
                // >::
								   default())
            }
        }
    }

    /// Min value of the keys (not! the min value in the vector)
    pub fn min(&self) -> i64 {
        match self {
            Likelihood1DContainer::Scalar(x) => x.min,
            Likelihood1DContainer::Matrix(x) => x.keys().copied().min().unwrap(),
        }
    }

    /// Max value of the keys (not! the max value in the vector)
    pub fn max(&self) -> i64 {
        match self {
            Likelihood1DContainer::Scalar(x) => x.max,
            Likelihood1DContainer::Matrix(x) => x.keys().copied().max().unwrap() + 1,
        }
    }

    /// Iterator, return (key, likelihood)
    pub fn iter(&self) -> impl Iterator<Item = (i64, Likelihood)> + '_ {
        match self {
            Likelihood1DContainer::Scalar(x) => Either::Left(
                x.iter()
                    .map(|(key, &value)| (key, Likelihood::Scalar(value))),
            ),
            Likelihood1DContainer::Matrix(x) => Either::Right(
                x.iter()
                    .map(|(&key, &value)| (key, Likelihood::Vector(Box::new(value)))),
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Likelihood2DContainer {
    Scalar(RangeArray2),
    Matrix(HashMap<(i64, i64), Box<Matrix16>>),
    // nohash_hasher::BuildNoHashHasher<(i64, i64)>>),
}

impl Likelihood2DContainer {
    pub fn dim(&self) -> ((i64, i64), (i64, i64)) {
        (self.min(), self.max())
    }

    /// Get the value at position `pos`
    pub fn get(&self, pos: (i64, i64)) -> Likelihood {
        match &self {
            Likelihood2DContainer::Scalar(x) => Likelihood::Scalar(x.get(pos)),
            Likelihood2DContainer::Matrix(x) => {
                if !x.contains_key(&pos) {
                    Likelihood::Matrix(Box::new(Matrix16::zeros()))
                } else {
                    Likelihood::Matrix(x.get(&pos).unwrap().clone())
                }
            }
        }
    }

    /// Add `likelihood` to the value at `pos`
    pub fn add_to(&mut self, pos: (i64, i64), likelihood: Likelihood) {
        match (self, likelihood) {
            (Likelihood2DContainer::Scalar(x), Likelihood::Scalar(ll)) => *x.get_mut(pos) += ll,
            (Likelihood2DContainer::Matrix(x), Likelihood::Matrix(ll)) => {
                x.entry(pos).and_modify(|v| **v += *ll).or_insert(ll);
            }
            _ => unimplemented!("Problem with add_to for Likelihood2DContainer"),
        }
    }

    /// Return a container with only 0 values
    pub fn zeros(starts: (i64, i64), ends: (i64, i64), dt: SequenceType) -> Likelihood2DContainer {
        match dt {
            SequenceType::Dna => Likelihood2DContainer::Scalar(RangeArray2::zeros((starts, ends))),
            SequenceType::Protein => Likelihood2DContainer::Matrix(HashMap::with_capacity(100)),
        }
    }

    /// Min value of the keys (not! the min value in the vector)
    pub fn min(&self) -> (i64, i64) {
        match self {
            Likelihood2DContainer::Scalar(x) => x.min,
            Likelihood2DContainer::Matrix(x) => (
                x.keys().copied().map(|u| u.0).min().unwrap(),
                x.keys().copied().map(|u| u.1).min().unwrap(),
            ),
        }
    }

    /// Max value of the keys (not! the max value in the vector)
    pub fn max(&self) -> (i64, i64) {
        match self {
            Likelihood2DContainer::Scalar(x) => x.max,
            Likelihood2DContainer::Matrix(x) => (
                x.keys().copied().map(|u| u.0).max().unwrap() + 1,
                x.keys().copied().map(|u| u.1).max().unwrap() + 1,
            ),
        }
    }

    /// Iterator, return (key, likelihood)
    pub fn iter(&self) -> impl Iterator<Item = (i64, i64, Likelihood)> + '_ {
        match self {
            Likelihood2DContainer::Scalar(x) => Either::Left(
                x.iter()
                    .map(|(x1, x2, &value)| (x1, x2, Likelihood::Scalar(value))),
            ),
            Likelihood2DContainer::Matrix(x) => Either::Right(
                x.iter()
                    .map(|(&key, value)| (key.0, key.1, Likelihood::Matrix(value.clone()))),
            ),
        }
    }

    /// Iterator, return (key, likelihood)
    pub fn iter_fixed_2nd(&self, dend: i64) -> impl Iterator<Item = (i64, Likelihood)> + '_ {
        match self {
            Likelihood2DContainer::Scalar(x) => Either::Left(
                x.iter_fixed_2nd(dend)
                    .map(|(x, &value)| (x, Likelihood::Scalar(value))),
            ),
            Likelihood2DContainer::Matrix(x) => Either::Right(
                x.iter()
                    .filter(move |(key, _)| key.1 == dend)
                    .map(|(&key, value)| (key.0, Likelihood::Matrix(value.clone()))),
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LikelihoodInsContainer {
    inner: [Likelihood2DContainer; 4],
}

impl LikelihoodInsContainer {
    pub fn dim(&self) -> ((i64, i64), (i64, i64)) {
        (self.min(), self.max())
    }

    pub fn get(&self, pos: (i64, i64), first_nucleotide: usize) -> Likelihood {
        self.inner[first_nucleotide].get(pos)
    }

    /// Add `likelihood` to the value at `pos`
    pub fn add_to(&mut self, pos: (i64, i64), first_nucleotide: usize, likelihood: Likelihood) {
        self.inner[first_nucleotide].add_to(pos, likelihood);
    }

    /// Return a container with only 0 values (for scalar likelihood)
    pub fn zeros(starts: (i64, i64), ends: (i64, i64), dt: SequenceType) -> LikelihoodInsContainer {
        LikelihoodInsContainer {
            inner: [
                Likelihood2DContainer::zeros(starts, ends, dt),
                Likelihood2DContainer::zeros(starts, ends, dt),
                Likelihood2DContainer::zeros(starts, ends, dt),
                Likelihood2DContainer::zeros(starts, ends, dt),
            ],
        }
    }

    /// Min value of the keys (not! the min value in the vector)
    pub fn min(&self) -> (i64, i64) {
        (
            self.inner.iter().map(|x| x.min().0).min().unwrap(),
            self.inner.iter().map(|x| x.min().1).min().unwrap(),
        )
    }

    /// Max value of the keys (not! the max value in the vector)
    pub fn max(&self) -> (i64, i64) {
        (
            self.inner.iter().map(|x| x.max().0).max().unwrap(),
            self.inner.iter().map(|x| x.max().1).max().unwrap(),
        )
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, i64, i64, Likelihood)> + '_ {
        self.inner
            .iter()
            .enumerate()
            .flat_map(|(i, a)| a.iter().map(move |(x, y, z)| (i, x, y, z)))
    }

    // pub fn iter_fixed_2nd(&self, dend: i64) -> impl Iterator<Item = (i64, &Likelihood)> + '_ {
    //     self.inner.iter().enu
    // }
}
