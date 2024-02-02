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
maturin develop --release -F py_binds,pyo3 --profile release
```

How to use:
-----------

Fast generation:
```py
import ihor
# Create generation model (once only)
gen = ihor.vdj.Generator(
"models/human_T_beta/model_params.txt",
"models/human_T_beta/model_marginals.txt",
"models/human_T_beta/V_gene_CDR3_anchors.csv",
"models/human_T_beta/J_gene_CDR3_anchors.csv")

# Generate productive amino-acid sequence
result = gen.generate(True) # False for unproductive
print(f"Full sequence: {result.full_seq}")
print(f"V gene: {result.v_gene}, J gene: {result.j_gene}")
print(f"CDR3: {result.cdr3_nt} {result.cdr3_aa}")
```


Inference:
```py
import ihor
from tqdm import tqdm

# load the model
model = ihor.vdj.Model.load_model("models/human_T_beta/model_params.txt",
"models/human_T_beta/model_marginals.txt",
"models/human_T_beta/V_gene_CDR3_anchors.csv",
"models/human_T_beta/J_gene_CDR3_anchors.csv")

# define parameters for the alignment and the inference
align_params = ihor.AlignmentParameters(min_score_v=0, min_score_j=0,max_error_d=100)
infer_params = ihor.InferenceParameters(min_likelihood=1e-400)

# read the file line by line and align each sequence
seq = []
with open('demo/murugan_naive1_noncoding_demo_seqs.txt') as f:
    for l in tqdm(f):
        s = model.align_sequence(l.strip(), align_params)
        r = model.infer(s, infer_params)
        print(r.likelihood)
```


Differences with IGoR:
- "dynamic programming" method, instead of summing over all events we first pre-compute over sum of events. This means that we can run it with undefined nucleotides like N (at least in theory, I need to add full support for these).
- The D gene alignment is less constrained

Limitations (I think also true for IGoR but not clear):
- Need to get rid of any primers/ends on the V gene side before running it
- The reads need to be long enough to fully cover the CDR3 (even when it's particularly long)
- still not sure if I should use initial_distribution for the insertion model


Programming stuff:
- Compile for python: `maturin develop --release -F py_binds,pyo3`
- I'm working on the web version on a different crate, importing the library, need to push that on git.

Things to do:
- test the inference
- add more tests
- deal with the "pgen with errors"
- deal with potential insertion in V/J alignment, remove the sequence from the inference if the insertion overlap with the delv range.
- write igor file, offer a json export
- deal with having a specific V gene (or a specific set of V gene) in the model
- support for VJ
- StaticEvent / GenEvent
- add an easier way to load the model (with just a keyword)
- fix the CDR3 problem

Bug:
- having spaces in the marginal file mess up the parsing silently, needs to be fixed
- if range_j / range_v / range_d doesn't match p_deld/p_delv/ p_delj in length, should complain
- generally should complain if the file is not exactly what is expected
- speed is way slower when compiled with python

Current status:
- speed is ok (50 seqs/s roughly ?). Could be slightly faster. I think some range should be replaced by iterator. Another big improvement would be to consider shorter V segment during the alignment.
- pgen works, but because I consider way more D gene alignment than Quentin some issue when endD and startD are really close to each other.
