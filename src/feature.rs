// This class define different type of Feature
// Feature are used during the expectation maximization process
// In short you need:
// - a function computing the likelihood of the feature `likelihood`
// - a function that allows to update the probability distribution
//   when new observations are made.
//   This update is done lazily, for speed reason we don't want to
//   redefine the function everytime.
// - so a function cleanup is going to return a new object with
//   the new probability distribution.
// This is the general idea, there's quite a lot of boiler plate
// code for the different type of categorical features (categorical features in
// 1d, in 1d but given another parameter, in 2d ...)

trait Feature<T> {
    fn dirty_update(&mut self, observation: T, likelihood: f64);
    fn likelihood(&self, observation: T);
    fn cleanup(&self) -> Result<Self, Box<dyn Error>>;
}

// One-dimensional categorical distribution
struct CategoricalFeature1 {
    probas: Array1<f64>,
    probas_dirty: Array1<f64>,
}

impl Feature<usize> for CategoricalFeature1 {
    fn dirty_update(&mut self, observation: usize, likelihood: f64) {
        probas_dirty[[observation]] += likelihood;
    }
    fn likelihood(&mut self, observation: usize) {
        probas[[observation]]
    }
    fn cleanup(&mut self) -> Result<CategoricalFeature1, Box<dyn Error>> {
        CategoricalFeature1::new(&self.probas_dirty)
    }
}

impl CategoricalFeature1 {
    fn new(probabilities: &Array1<f64>) -> Result<CategoricalFeature1, Box<dyn Error>> {
        CategoricalFeature1 {
            probas: probabilities.normalize_distribution(None)?,
            probas_dirty: Array1::<f64>::zeros(self.probabilities.dim()),
        }
    }
}

// One-dimensional categorical distribution, given one external parameter
struct CategoricalFeature1g1 {
    probas: Array2<f64>,
    probas_dirty: Array2<f64>,
}

impl Feature<(usize, usize)> for CategoricalFeature1g1 {
    fn dirty_update(&mut self, observation: (usize, usize), likelihood: f64) {
        probas_dirty[[observation.0, observation.1]] += likelihood;
    }
    fn likelihood(&mut self, observation: usize) {
        probas[[observation.0, observation.1]]
    }
    fn cleanup(&mut self) -> Result<CategoricalFeature1g1, Box<dyn Error>> {
        CategoricalFeature1g1::new(&self.probas_dirty)
    }
}

impl CategoricalFeature1g1 {
    fn new(probabilities: &Array2<f64>) -> Result<CategoricalFeature1g1, Box<dyn Error>> {
        CategoricalFeature1g1 {
            probas: probabilities.normalize_distribution(Some(Axis(0)))?,
            probas_dirty: Array2::<f64>::zeros(self.probabilities.dim()),
        }
    }
}

// Two-dimensional categorical distribution
struct CategoricalFeature2 {
    probas: Array2<f64>,
    probas_dirty: Array2<f64>,
}

impl Feature<(usize, usize)> for CategoricalFeature2 {
    fn dirty_update(&mut self, observation: (usize, usize), likelihood: f64) {
        probas_dirty[[observation.0, observation.1]] += likelihood;
    }
    fn likelihood(&mut self, observation: usize) {
        probas[[observation.0, observation.1]]
    }
    fn cleanup(&mut self) -> Result<CategoricalFeature2, Box<dyn Error>> {
        CategoricalFeature2::new(&self.probas_dirty)
    }
}

impl CategoricalFeature2 {
    fn new(probabilities: &Array2<f64>) -> Result<CategoricalFeature2, Box<dyn Error>> {
        CategoricalFeature2 {
            probas: probabilities
                .normalize_distribution(Some(Axis(0)))?
                .normalize_distribution(Some(Axis(1)))?,
            probas_dirty: Array2::<f64>::zeros(self.probabilities.dim()),
        }
    }
}

// Two-dimensional categorical distribution, given one external parameter
struct CategoricalFeature2g1 {
    probas: Array3<f64>,
    probas_dirty: Array3<f64>,
}

impl Feature<(usize, usize, usize)> for CategoricalFeature2g1 {
    fn dirty_update(&mut self, observation: (usize, usize, usize), likelihood: f64) {
        probas_dirty[[observation.0, observation.1, observation.2]] += likelihood;
    }
    fn likelihood(&mut self, observation: usize) {
        probas[[observation.0, observation.1, observation.2]]
    }
    fn cleanup(&mut self) -> Result<CategoricalFeature2g1, Box<dyn Error>> {
        CategoricalFeature2g1::new(self.probas_dirty)
    }
}

impl CategoricalFeature1g1 {
    fn new(probabilities: Array2<f64>) -> Result<CategoricalFeature2g1, Box<dyn Error>> {
        CategoricalFeature2g1 {
            probas: probabilities
                .normalize_distribution(Some(Axis(0)))?
                .normalize_distribution(Some(Axis(1)))?,
            probas_dirty: Array3::<f64>::zeros(self.probabilities.dim()),
        }
    }
}

// Markov chain structure for Dna insertion
struct MarkovFeature {
    initial_distribution: Array1<f64>,
    transition_matrix: Array2<f64>,
    initial_distribution_dirty: Array1<f64>,
    transition_matrix_dirty: Array2<f64>,
}

impl Feature<&Dna> for MarkovFeature {
    fn dirty_update(&mut self, observation: &Dna, likelihood: f64) {
        if observation.len() == 0 {
            return;
        }
        self.initial_distribution_dirty[[NUCLEOTIDES_INV[&observation.seq[0]]]] += likelihood;
        for ii in 1..observation.len() {
            self.transition_matrix_dirty[[
                NUCLEOTIDES_INV[&observation.seq[ii - 1]],
                NUCLEOTIDES_INV[&observation.seq[ii]],
            ]] += likelihood;
        }
    }
    fn likelihood(&mut self, observation: &Dna) -> f64 {
        if d.len() == 0 {
            return 1.;
        }
        let mut proba = self.initial_distribution[NUCLEOTIDES_INV[&observation.seq[0]]];
        for ii in 1..d.len() {
            proba *= self.transition_matrix[[
                NUCLEOTIDES_INV[&observation.seq[ii - 1]],
                NUCLEOTIDES_INV[&observation.seq[ii]],
            ]];
        }
        proba
    }
    fn cleanup(&mut self) -> Result<MarkovFeature, Box<dyn Error>> {
        MarkovFeature::new(&self.initial_distribution, &self.transition_matrix)
    }
}

impl MarkovFeature {
    fn new(
        initial_distribution: &Array1<f64>,
        transition_matrix: &Array2<f64>,
    ) -> Result<MarkovFeature, Box<dyn Error>> {
        MarkovFeature {
            transition_matrix: transition_matrix.normalize_distribution(Some(Axis(1)))?,
            initial_distribution: initial_distribution.normalize_distribution(None)?,
            transition_matrix_dirty: Array2::<f64>::zeros(transition_matrix.dim()),
            initial_distribution_dirty: Array1::<f64>::zeros(initial_distribution.dim()),
        }
    }
}

// Most basic error model
struct ErrorPoisson {
    error_rate: f64,
    lookup_table: Vec<f64>,
    total_probas_dirty: f64, // useful for dirty updating
    total_errors_dirty: f64, // same
}

impl ErrorPoisson {
    fn new(error_rate: f64, min_error_rate: f64) -> ErrorPoisson {
        let mut e = ErrorPoisson {
            error_rate: error_rate,
            min_error_rate: min_error_rate,
            total_probas_dirty: 0.,
            total_errors_dirty: 0.,
            lookup_table: make_lookup_table(error_rate, min_error_rate),
        };
        e
    }

    fn make_lookup_table(error_rate: f64, min_error_rate: f64) -> Vec<f64> {
        let mut lookup_table = Vec::<f64>::new();
        let mut prob = (-error_rate).exp();
        let mut nb = 0;
        loop {
            lookup_table.push(prob);
            nb += 1;
            prob *= error_rate / nb;
            if prob < min_error_rate {
                break;
            }
        }
        lookup_table
    }
}

impl Feature<usize> for ErrorPoisson {
    fn dirty_update(&mut self, observation: usize, likelihood: f64) {
        total_probas += likelihood;
        total_errors += likelihood * nb_errors;
    }
    fn likelihood(&mut self, observation: usize) -> f64 {
        if observation >= self.lookup_table.len() {
            0.
        } else {
            self.lookup_table[[observation]]
        }
    }
    fn cleanup(&mut self) -> Result<ErrorPoisson, Box<dyn Error>> {
        // estimate the error_rate from the dirty estimates
        Ok(ErrorPoisson {
            error_rate: total_errors / total_probas,
            lookup_table: make_lookup_table(total_errors / total_probas, min_error_rate),
            min_error_rate: min_error_rate,
            total_probas_dirty: 0.,
            total_errors_dirty: 0.,
        })
    }
}
