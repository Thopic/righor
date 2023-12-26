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
gen = ihor.GeneratorVDJ( # ihor.GeneratorVJ
"models/human_T_beta/model_params.txt",
"models/human_T_beta/model_marginals.txt",
"models/human_T_beta/V_gene_CDR3_anchors.csv",
"models/human_T_beta/J_gene_CDR3_anchors.csv")

# Generate productive amino-acid sequence
result = gen.generate(True) # False for unproductive
print(f"Full sequence: {result.full_seq}")
print(f"V gene: {result.v_name}, J gene: {result.j_name}")
print(f"CDR3: {result.cdr3_nt} {result.cdr3_aa}")
```


Inference:
----------

Limitations:
- Need to get rid of any primers/ends on the V gene side.
- The reads need to be long enough to fully cover the CDR3 (even when it's particularly long)

```py
import ihor
from tqdm import tqdm

model = ihor.vdj.Model.load_model("models/human_T_beta/model_params.txt",
	"models/human_T_beta/model_marginals.txt",
	"models/human_T_beta/V_gene_CDR3_anchors.csv",
	"models/human_T_beta/J_gene_CDR3_anchors.csv")

align_params = ihor.AlignmentParameters(min_score_v= 40, min_score_j= 10, max_error_d=10)

seq = []
with open('demo/murugan_naive1_noncoding_demo_seqs.txt') as f:
	for l in tqdm(f):
		seq += [model.align_sequence(ihor.Dna.from_string(l.strip()), align_params)]

infer_params = ihor.InferenceParameters(min_likelihood=1e-40, min_likelihood_error=1e-60)

```

Programming stuff:
Some unneeded duplicated code, mostly because pyo3 not compatible with templates
open_blas is a pain in the ass, all that for one not-very-important diagonalisation.
