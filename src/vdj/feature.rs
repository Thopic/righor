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

// So if we don't have access to the D alignment yet, we can still pre-compute
// P(delV < delVmax) and P(delV = x) <- this one is trivial

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

// Example
fn infer() {
    for val in V_Alignments {
        for dal in DAlignments {
            for jal in J_Alignments {
                let likelihood_estimation = likelihood_estimate(val, dal, jal);
                if likelihood_estimation > min_likelihood {
                    infer_fixed_alignment(&val, &dal, &jal)
                }
            }
        }
    }
}

fn infer_fixed_alignment(val: &VJAlignment, jal: &VJAlignment, dal: &DAlignment) {
    let range_del_v = if val.end > cmp::min(jal.start, dal.start) {
        (cmp::min(jal.start, dal.start), val.end)
    } else {
        (val.end, val.end)
    };

    let range_del_j = ();
    let range_del_d = ();

    for del_d in range_del_v {
        let ld = agg_feat_d.likelihood(ld, range_del_d);
        for del_j in range_del_j {
            let lj = agg_feat_j.likelihood(lj, range_del_j);
            for del_d in range_del_d {
                let ld = agg_feat_j.likelihood(lj, range_del_j);
                // dirty update
            }
        }
    }
    // clean up
}

struct FeatureV {
    pub v: &CategoricalFeature1,
    pub delv: &CategoricalFeature1g1,
    pub ins: InsertionFeature,
    pub v_alignment: VJAlignment,

    probas_deletions_lower_than_dirty: Vec<f64>,
    log_likelihood_deletions_lower_than: Vec<f64>,
}

impl AggregatedFeatureV {
    // Create the object and precompute the values
    fn new() -> FeatureV {
        for min_del_v in enumerate(probas_deletions_lower_than_dirty) {
            for del_v in min_del_v..max_del_v {
                let ins_v = seq[..del_v];
                probability += feat.delv.likelihood(del_v)
                    * feat.ins.likelihood(&ins_v)
                    * feat.error.likelihood(v.nb_errors(del_v))
            }
        }
    }

    /// Return log(P(delV > del_v | v alignment))
    fn log_likelihood_lower_than(&self, del_v: usize) {}

    /// Return log(P(delV == del_v | v alignment))
    fn log_likelihood_equal(&self, del_v: usize) {}

    /// Update both the direct probabilities (P(delV == del_V)
    /// and the more complicated ones (P(delV < del_V))
    fn dirty_update(&mut self, event: &e, likelihood: f64) {
        if condition_one {
        } else {
            dirty_probas_deletions_lower_than[e.righthing] += likelihood;
        }
    }

    /// Transform the non direct distribution into the original
    /// distribution
    fn cleanup(
        &self,
        &mut ins: InsertionFeature,
        &mut del: DeletionFeature,
        &mut error: ErrorFeature,
    ) {
        for min_del_v in enumerate(probas_deletions_lower_than_dirty) {
            for del_v in min_del_v..max_del_v {
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
