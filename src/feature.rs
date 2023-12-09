// A sequence is a collection of feature, each feature has some probability
// When we iterate on the model we update the probability of each feature


pub struct Feature {
    pub v_gene: (usize, usize),
    pub j_gene: (usize, usize),
    pub d_gene: (usize, usize),
    pub del_v: usize,
    pub del_j: usize,
    pub del_d_3: usize,
    pub del_d_5: usize,
}


// define a pre-sequence, with list of plausible V-gene & plausible alignments



// infer_feature(seq)
// given a sequence, try to compute all possible deletion / insertion / D genes combination + proba.
// Given the current proba model. Return a list of Ns most likely features with probas

// Given a recombination event I need a quick way to associate proba to that feature
// An event is
// V, D, J genes and their position
// delV, delDs, delJs
// Probability involve the full model

// I want to define a function probability_feature_given_sequence(f: Feature, s: Sequence, m: Model) that returns the
// probability of a given feature [plausible to do dynamical programming and return the proba of multiple features ?]

// Then I need to sum on these features over all sequences to get a marginal.

// Note that the insertion process is largely disconnected from the rest, so we can infer it on its own (ie, compute the marginal independantly later on).

//

// For example, seems do-able for del / ins mechanisms. Summing makes sense, but in most case I actually need to count. Fck.
// At least it'll be simpler to code


fn infer_features(sequence: Sequence, model: Model) {
    for v in sequence.v_genes {
	for j in sequences.j_genes {
	    for d in sequences.d_genes {
		logp = model.log_proba_vdj() + model.log_proba




	    }
	}
    }


}
