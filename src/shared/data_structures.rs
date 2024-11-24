//! Contains data structures (`RangeArray`)
//! `RangeArray` are array structures (similar to `ndarray`)
//! containing f64 indexed by i64, with fast access

fn max_vector(arr: &[f64]) -> Option<f64> {
    arr.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).copied()
}

/// Implement an array structure containing f64, indexed by min..max where min/max are i64
#[derive(Default, Clone, Debug)]
pub struct RangeArray1 {
    pub array: Vec<f64>,
    pub min: i64,
    pub max: i64, // other extremity of the range (min + array.len())
}

impl RangeArray1 {
    // iterate over index & value
    pub fn iter(&self) -> impl Iterator<Item = (i64, &f64)> + '_ {
        (self.min..self.max).zip(self.array.iter())
    }

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
        debug_assert!(idx >= self.min && idx < self.max);
        //unsafe because improve perf
        //        unsafe {
        *self.array.get((idx - self.min) as usize).unwrap()
        //}
    }

    pub fn max_value(&self) -> f64 {
        max_vector(&self.array).unwrap()
    }

    pub fn dim(&self) -> (i64, i64) {
        (self.min, self.max)
    }

    pub fn len(&self) -> usize {
        (self.max - self.min) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.max == self.min
    }

    pub fn zeros(range: (i64, i64)) -> RangeArray1 {
        RangeArray1 {
            min: range.0,
            max: range.1,
            array: vec![0.; (range.1 - range.0) as usize],
        }
    }

    pub fn constant(range: (i64, i64), cstt: f64) -> RangeArray1 {
        RangeArray1 {
            min: range.0,
            max: range.1,
            array: vec![cstt; (range.1 - range.0) as usize],
        }
    }

    pub fn get_mut(&mut self, idx: i64) -> &mut f64 {
        debug_assert!(idx >= self.min && idx < self.max);
        //unsafe because improve perf
        //        unsafe {
        self.array.get_mut((idx - self.min) as usize).unwrap()
        //}
    }

    pub fn mut_map<F>(&mut self, mut f: F)
    where
        F: FnMut(f64) -> f64,
    {
        self.array.iter_mut().for_each(|x| *x = f(*x));
    }
}

/// Implement an array structure containing f64, indexed by min..max where min/max are i64
#[derive(Default, Clone, Debug)]
pub struct RangeArray3 {
    pub array: Vec<f64>,
    pub min: (i64, i64, i64),
    pub max: (i64, i64, i64),
    nb0: i64,
    nb1: i64,
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
        let nb0 = max.0 - min.0;
        let nb1 = max.1 - min.1;

        let mut array = vec![0.; (nb0 * nb1 * (max.2 - min.2)) as usize];
        for ((i0, i1, i2), value) in values {
            array[(i0 - min.0) as usize
                + ((i1 - min.1) * nb0) as usize
                + ((i2 - min.2) * nb1 * nb0) as usize] += value;
        }
        RangeArray3 {
            array,
            min,
            max,
            nb0,
            nb1,
        }
    }

    pub fn max_value(&self) -> f64 {
        max_vector(&self.array).unwrap()
    }

    // return min
    pub fn lower(&self) -> (i64, i64, i64) {
        self.min
    }

    // return max + 1
    pub fn upper(&self) -> (i64, i64, i64) {
        self.max
    }

    pub fn get(&self, idx: (i64, i64, i64)) -> f64 {
        debug_assert!(
            idx.0 >= self.min.0
                && idx.0 < self.max.0
                && idx.1 >= self.min.1
                && idx.1 < self.max.1
                && idx.2 >= self.min.2
                && idx.2 < self.max.2
        );
        //        unsafe {
        *self
            .array
            .get(
                (idx.0 - self.min.0) as usize
                    + ((idx.1 - self.min.1) * self.nb0) as usize
                    + ((idx.2 - self.min.2) * self.nb1 * self.nb0) as usize,
            )
            .unwrap()
        //        }
    }

    pub fn get_mut(&mut self, idx: (i64, i64, i64)) -> &mut f64 {
        debug_assert!(
            idx.0 >= self.min.0
                && idx.0 < self.max.0
                && idx.1 >= self.min.1
                && idx.1 < self.max.1
                && idx.2 >= self.min.2
                && idx.2 < self.max.2
        );
        //        unsafe {
        self.array
            .get_mut(
                (idx.0 - self.min.0) as usize
                    + ((idx.1 - self.min.1) * self.nb0) as usize
                    + ((idx.2 - self.min.2) * self.nb1 * self.nb0) as usize,
            )
            .unwrap()
        //        }
    }

    pub fn dim(&self) -> ((i64, i64, i64), (i64, i64, i64)) {
        (self.min, self.max)
    }

    pub fn zeros(range: ((i64, i64, i64), (i64, i64, i64))) -> RangeArray3 {
        RangeArray3 {
            min: range.0,
            max: range.1,
            nb0: range.1 .0 - range.0 .0,
            nb1: range.1 .1 - range.0 .1,
            array: vec![
                0.;
                ((range.1 .0 - range.0 .0) * (range.1 .1 - range.0 .1) * (range.1 .2 - range.0 .2))
                    as usize
            ],
        }
    }

    pub fn constant(range: ((i64, i64, i64), (i64, i64, i64)), cstt: f64) -> RangeArray3 {
        RangeArray3 {
            min: range.0,
            max: range.1,
            nb0: range.1 .0 - range.0 .0,
            nb1: range.1 .1 - range.0 .1,
            array: vec![
                cstt;
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

    pub fn iter(&self) -> impl Iterator<Item = (i64, i64, i64, &f64)> + '_ {
        self.array.iter().enumerate().map(|(idx, v)| {
            (
                (idx as i64) % self.nb0 + self.min.0,
                ((idx as i64) / self.nb0) % self.nb1 + self.min.1,
                (idx as i64) / (self.nb1 * self.nb0) + self.min.2,
                v,
            )
        })
    }
}

/// Implement an array structure containing f64, indexed by min..max where min/max are i64
#[derive(Default, Clone, Debug)]
pub struct RangeArray2 {
    pub array: Vec<f64>,
    pub min: (i64, i64),
    pub max: (i64, i64),
    nb0: i64,
}

impl RangeArray2 {
    // iterate over indexes and values
    pub fn iter(&self) -> impl Iterator<Item = (i64, i64, &f64)> + '_ {
        self.array.iter().enumerate().map(|(idx, v)| {
            (
                (idx as i64) % self.nb0 + self.min.0,
                (idx as i64) / self.nb0 + self.min.1,
                v,
            )
        })
    }

    // iterate over indexes and values
    pub fn iter_fixed_2nd(&self, v2: i64) -> impl Iterator<Item = (i64, &f64)> + '_ {
        self.array
            [((v2 - self.min.1) * self.nb0) as usize..((v2 - self.min.1 + 1) * self.nb0) as usize]
            .iter()
            .enumerate()
            .map(|(idx, v)| ((idx as i64 + self.min.0), v))
    }

    pub fn new(values: &Vec<((i64, i64), f64)>, cstt: f64) -> RangeArray2 {
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
        let nb0 = max.0 - min.0;

        let mut array = vec![cstt; (nb0 * (max.1 - min.1)) as usize];
        for ((i0, i1), value) in values {
            array[(i0 - min.0) as usize + ((i1 - min.1) * nb0) as usize] += value;
        }
        RangeArray2 {
            array,
            min,
            max,
            nb0,
        }
    }

    pub fn max_value(&self) -> f64 {
        max_vector(&self.array).unwrap()
    }

    pub fn get(&self, idx: (i64, i64)) -> f64 {
        debug_assert!(
            idx.0 >= self.min.0 && idx.0 < self.max.0 && idx.1 >= self.min.1 && idx.1 < self.max.1
        );
        //        unsafe {
        *self
            .array
            .get((idx.0 - self.min.0 + (idx.1 - self.min.1) * self.nb0) as usize)
            .unwrap()
        //      }
    }

    pub fn get_mut(&mut self, idx: (i64, i64)) -> &mut f64 {
        debug_assert!(
            idx.0 >= self.min.0 && idx.0 < self.max.0 && idx.1 >= self.min.1 && idx.1 < self.max.1
        );
        //    unsafe {
        self.array
            .get_mut((idx.0 - self.min.0 + (idx.1 - self.min.1) * self.nb0) as usize)
            .unwrap()
        //  }
    }

    pub fn dim(&self) -> ((i64, i64), (i64, i64)) {
        (self.min, self.max)
    }

    // return min
    pub fn lower(&self) -> (i64, i64) {
        self.min
    }

    // return max + 1
    pub fn upper(&self) -> (i64, i64) {
        self.max
    }

    pub fn zeros(range: ((i64, i64), (i64, i64))) -> RangeArray2 {
        RangeArray2 {
            min: range.0,
            max: range.1,
            nb0: range.1 .0 - range.0 .0,
            array: vec![0.; ((range.1 .0 - range.0 .0) * (range.1 .1 - range.0 .1)) as usize],
        }
    }

    pub fn constant(range: ((i64, i64), (i64, i64)), cstt: f64) -> RangeArray2 {
        RangeArray2 {
            min: range.0,
            max: range.1,
            nb0: range.1 .0 - range.0 .0,
            array: vec![cstt; ((range.1 .0 - range.0 .0) * (range.1 .1 - range.0 .1)) as usize],
        }
    }

    pub fn mut_map<F>(&mut self, mut f: F)
    where
        F: FnMut(f64) -> f64,
    {
        self.array.iter_mut().for_each(|x| *x = f(*x));
    }
}

// /// Specific structure for insertion
// #[derive(Default, Clone, Debug)]
// pub struct InsertionArray {
//     pub array: Vec<f64>,
//     pub min: (i64, i64),
//     pub max: (i64, i64),
//     nb0: i64,
// }
