// This is an attempt to do some dynamic programming on Igor inference
// The basic idea:
//
// Given an alignment V D
//           0
// V: ------]
// D:            [------
// We can compute all the deletions probabilities for V by summing over all del_V
// P(delV <= 0) = \sum_{del_V} P(del_V | V) * P(error = errors_left[:-delV]) * P(insertions [-delV:])
// to compute the full expression we then do P(V < 0) * P(D >= 4) * P(insertion_left)

// In the more likely event where both V and D overlap we become more careful:

// idx:   0    5  8
// V:     -------]
// D:           [----------
// Cutting over the easy values is only allowed outside of the shared alignment
// So instead of computing Σ P(delV = x) * P(delD = y) * P(ins) for all x (and inferring the full insVD each time),
// We compute P(V < 6) * (P(D=6)*ins(6..6) +P(D=7) * ins(6..7) + P(D > 7) * ins(6..8)) + P(V = 6)*(P(D=7)*ins(7..7) + P(D>7)*ins(7..8)) + P(V=7) * P(D>7)*ins(8..8)
// Other option P(V < 8) * P(D > 7) + P(V < 7) * P(D == 7) + P(V < 6) * P(D == 6)
// Maybe a good choice too, I'm just a bit afraid of potential V / J overlap (but maybe I'm fine if I define an order of action V -> J -> D
// Example in that last scenario
// I:         8
// V: --------]
// D:   [-------]
// J:       [----------
// Then: P(V < 8)*P(D5>=8, D3 <=8) * P(J >=8) + P(V < 8) * P(D5>=8, D3 == 9) * P(J > 9) + P(V < 8) * P(D5 >= 8, D3 == 10) * P(J > 10)
//     + P(V < 7)*P(D5>=7, D3 <=7) * P(J == 7) + P(V < 7)*P(D5>=7, D3 == 8) * P(J == 8) + ...
// which obviously strongly cut off the number of sums and loop. But needs to be *very* careful to not count things twice...

// The sum will be transformed in the following way:

// \sum_{d_v} \sum_{d_j \in rj(d_v)} \sum_{d_{d5} \in r5(d_v, d_j)} \ sum_{d_{d3} \in r3(d_v, d_j)} p(d_v) * p(d_j) * p(d_{d3, d5}) * ins(v->d) * ins(d->j)
// \sum_{max_d_v}  \sum_{d_v =0}^{max_d_v} p(d_v = max_d_v)

// So if we don't have access to the D alignment yet, we can still pre-compute
// P(delV < delVmax) and P(delV = x) <- this one is trivial
// Basically everything can be pre-computed (given the features & the sequence)

// This works for evaluation, but what do we do for inference ?
// Each time we get one likelihood we feed one of the probability bins, (P(V < 6), P(V=6), P(V=7) dans l'exemple précédent)
// Then once all the scenarios have been accounted for we update the probabilities of the underlying distributions
// By redoing the exact same computations that we did when we computed the sum probabilities in the first place.

// This is not going to be easy to implement though

// Another potential issue is how to deal with "most likely". Currently I just iterate over the specific event, rather than group
// of event. Getting the most likely event is relatively painless, but anything more complicated is going to be a world of pain.
// I need to keep the OG method somewhere then ? Or just keep the most likely event ? Note that this doesn't affect the position
// (or ranking) of V/D/J though. One option would be to run the OG algorithm over this specific V/D/J (I should have it somewhere
// in any case to check that everything works well).

// For the D gene, same strat, except that there will be two dels, on both side, to take care off (so Array2 ?)

// Practically, how am I going to deal with that loop ?
// Create a struct with two states del_v and <= del_v and make a function to iterate over these
// For D, four states [deld5, deld3], [>= deld5, deld3], [deld5, <= deld3], [>= deld5, <= deld3]

// Doesn't work, because of insertion length linking everything together anyhow...

// Example
fn infer() {
    for val in V_Alignments {
        for dal in DAlignments {
            for jal in VJ_Alignments {
                let likelihood_estimation = likelihood_estimate(val, dal, jal);
                if likelihood_estimation > min_likelihood {
                    infer_fixed_alignment(&val, &dal, &jal)
                }
            }
        }
    }
}

fn infer_given_alignment(val: &VJAlignment, jal: &VJAlignment, dal: &DAlignment) {
    // end is always n+1, so
    //          endV
    //           v
    // V: ------]

    let aggreg_vs = Vec::new();
}

struct FeatureV {
    ll_deletions_lower_than_dirty: Vec<f64>,
    ll_deletions_lower_than: Vec<f64>,
}

impl AggregatedFeatureV {
    // Create the object and precompute the values
    fn new(
        val: &VJAlignment,
        dal: &DAlignment,
        jal: &VJAlignment,
        seq: &Dna,
        features: &Features,
    ) -> AggregatedFeatureV {
        let end_v = val.end_gene;
        let start_d = dal.pos;
        let end_d = dal.pos + dal.len_d;
        let start_j = jal.start_gene;
        let ll_deletions_lower_than = Vec::new();
        for min_del_v in 0..end_v - cmp::min(start_d, start_j) {
            ll_deletions_lower_than.push(pre_compute_ll_lower_than(min_del_v, val, seq, features));
        }
        AggregatedFeatureV {
            ll_more_deletions_than,
            dirty_ll_more_deletions_than: vec![0., ll_deletions_lower_than.dim()],
        }
    }

    /// Pre-compute P(delV > min_del_v | alignment, sequence)
    fn pre_compute_ll_lower_than(
        &self,
        min_del_v: usize,
        val: &VJAlignment,
        seq: &Dna,
        features: &Features,
    ) -> proba {
        let proba = 0;
        for del_v in min_del_v..features.deletion.dim() {
            proba += features.deletion.likelihood(del_v)
                * features
                    .insertion
                    .likelihood(seq[val.end_seq - del_v..val.end_seq - min_del_v])
                * features
                    .error
                    .likelihood(val.nb_errors(del_v), val.length_with_deletion(del_V))
        }
        proba
    }

    /// Return log(P(delV > del_v | alignment))
    fn log_likelihood_more_than(&self, del_v: usize) {
        self.ll_more_deletions_lower_than[del_v]
    }

    /// Update the probabilities P(delV > del_V | alignment)
    fn dirty_update(&mut self, event: &e, likelihood: f64) {
        self.dirty_ll_more_deletions_than[e.delv] += likelihood;
    }

    /// Transform the non direct distribution into the original
    /// distribution
    fn cleanup(&self, &mut features: Features) {
        for min_del_v in enumerate(probas_deletions_lower_than_dirty) {
            for del_v in min_del_v..features.deletion.dim() {
                let ins_v = seq[..del_v];
                let probability = feat.delv.likelihood(del_v)
                    * feat.ins.likelihood(&ins_v)
                    * feat.error.likelihood(v.nb_errors(del_v));
                ins.dirty_update(
                    &ins_v,
                    dirty_probas_deletions_lower_than[min_del_v] * probability / tot_probability,
                );
                del.dirty_update(
                    del_v,
                    dirty_probas_deletions_lower_than[min_del_v] * probability / tot_probability,
                );
                error.dirty_update(v.nb_errors(del_v));
            }
        }
    }
}
