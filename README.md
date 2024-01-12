# IHOR

Install rust (potentially slow):
--------------------------------

``` sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Install the library:
--------------------

In the git folder:
``` sh
pip install maturin
maturin develop --release
```

How to use:
-----------

Fast generation:
```py
# Create generation model (once only)
gen = ihor.vdj.Generator(
"models/human_T_beta/model_params.txt",
"models/human_T_beta/model_marginals.txt",
"models/human_T_beta/V_gene_CDR3_anchors.csv",
"models/human_T_beta/J_gene_CDR3_anchors.csv")

# For a VJ model
# gen = ihor.vj.Generator(
# "models/human_T_alpha/model_params.txt",
# "models/human_T_alpha/model_marginals.txt",
# "models/human_T_alpha/V_gene_CDR3_anchors.csv",
# "models/human_T_alpha/J_gene_CDR3_anchors.csv")


# Generate productive amino-acid sequence
result = gen.generate(True) # False for unproductive
print(f"Full sequence: {result.full_seq}")
print(f"V gene: {result.v_gene}, J gene: {result.j_gene}")
print(f"CDR3: {result.cdr3_nt} {result.cdr3_aa}")
```


Inference:
----------


```py
import ihor
from tqdm import tqdm

# load the model
model = ihor.vdj.Model.load_model("models/human_T_beta/model_params.txt",
	"models/human_T_beta/model_marginals.txt",
	"models/human_T_beta/V_gene_CDR3_anchors.csv",
	"models/human_T_beta/J_gene_CDR3_anchors.csv")

# define parameters for the alignment and the inference
align_params = ihor.AlignmentParameters(min_score_v= 40, min_score_j= 10, max_error_d=10)
infer_params = ihor.InferenceParameters(min_likelihood=1e-40, min_likelihood_error=1e-60)

# read the file line by line and align each sequence
seq = []
with open('demo/murugan_naive1_noncoding_demo_seqs.txt') as f:
	for l in tqdm(f):
		seq += [model.align_sequence(ihor.Dna.from_string(l.strip()), align_params)]

nb_rounds = 5
# Expectation-Maximization
for _ in range(nb_rounds):
	features = []
	for s in tqdm(seq):
		# infer the feature of each sequence
		features += [model.infer_features(s, infer_params)]
	# then compute the average
	new_feat = ihor.vdj.Features.average(features)
	# update the model
	model.update(new_feat)


```

Limitations (I think also true for IGoR but not clear):
- Need to get rid of any primers/ends on the V gene side.
- The reads need to be long enough to fully cover the CDR3 (even when it's particularly long)
- still not sure if I should use initial_distribution for the insertion model


Programming stuff:
- Some unneeded duplicated code, mostly because pyo3 not compatible with templates + need to be removed for wasm compilation
- open_blas is a pain in the ass, takes forever to compile, all that for one not-very-important diagonalisation -> now it's removed.
- log always means log2. ln is neperian.


Things to do:
- add more tests (interaction insertion + deletion, more than one v gene)
- Error model is wrong as defined rn
- deal with potential insertion in V/J alignment, remove the sequence from the inference if the insertion overlap with the delv range.
- if range_j / range_v / range_d doesn't match p_deld/p_delv/ p_delj in length, should complain

Current status:
- speed is ok. Could be slightly faster. My improvement have not been crazy, so I think I'm reaching a wall. Obvious changes would be to add some unsafe (beuh). It's at Igor speed now, so it's fine. alignment procedure is a bit slow too (but that's basically library code). I could check insertion a bit less & play a bit with extract_padded_sequence. But all together, I think it's in a pretty good state. Some easy fix: stop using Array so much, replace a lot of loop with iterators, maybe bit of unsafe.
- the inference doesn't quite work. Probably a few small problems here and there, need to check with the test. deletions without insertions maaaaaybe work. error kind of work but don't quite go to the right value.
- generally needs a lot more testing
- Consider just the best alignment, rather than all the alignment, to decide the range of D ? With some range ?
- Also improve the alignment, it's waaay too slow. We don't expect that many insertions. Start with aligning a fairly conserved subset of Vs, then try to align the other ones, following the same structure ?
- I've just realized that igor is P(V,D,J) while olga is P(V)P(D,J)... Need to add some flexibility in reading the model. Let's always work in the (more complex) P(V,D,J) case, but still give access to P(V) and P(D,J).


Tests:
- without deletions the inference seems to go well, including with the error rate.
- not the case with deletions. Even without insertions.
- Interestingly, while the error rate seemed to work with the full sequence it drastically fall when just CDR3. Wait.
