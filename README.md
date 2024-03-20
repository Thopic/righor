# RIGHOR

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
import righor
# Create generation model (once only)
gen = righor.vdj.Generator(
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
import righor
from tqdm import tqdm

# load the model
model = righor.vdj.Model.load_model("models/human_T_beta/model_params.txt",
"models/human_T_beta/model_marginals.txt",
"models/human_T_beta/V_gene_CDR3_anchors.csv",
"models/human_T_beta/J_gene_CDR3_anchors.csv")

# define parameters for the alignment and the inference
align_params = righor.AlignmentParameters(min_score_v=0, min_score_j=0,max_error_d=100)
infer_params = righor.InferenceParameters(min_likelihood=1e-400)

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
- I'm working on the web version on a different crate, importing the library, need to push that on git.
- python version: also a different crate now (will maybe loop it back in)
- when adding a model, add it to "models.json". First model in a category is the default model. Each field is one independant model. The elements in chain and species should always be lower-case.


Things to do:
- test the inference in detail
- add more tests
- deal with the "pgen with errors"
- deal with potential insertion in V/J alignment, remove the sequence from the inference if the insertion overlap with the delv range.
- test the restricted V gene option for generation.
- write igor file, offer a json export
- StaticEvent / GenEvent
- modify the way I deal with added error (make it cleaner, with a "ErrorDistribution" thing or smt)
- deal with amino-acid and generic "undefined" stuff.
Strat: define an extended Dna object that the alignment can deal with +
define the insertion thing so that it can deal with that
This second one is slightly a pain (the first one too ? No it's fine, just a bit longer to deal with).
I would need to add sums here and there, nothing impossible, but slightly more a pain. In short some position must be linked, this will complexify quite a bit the definition of Dna (more precisely this will be a new class). So
UndefinedDna would contains for each position a vec/array of bytes and a int giving the positions they're connected with  (just need two options for everything). This is very specific to the aa case, but why should I care. A bit complicated rn, leaving it for later.
- improve alignment so that it can deal with potential indels.
- add simpler inference (without full VDJ, without V-J...)
- publish cargo package
- json export and loading
- run cargo clippy
- clean up gen event / static event if possible.
- make it work with CDR3 + V gene + J gene (require implementing some of the python function in rust)


TODO before v0.2:
- publish pip package
- make a python notebook for example with: load model, align sequences, display aligned sequences, evaluate, display evaluate (incl. features), infer model, display inferred model.
- use pgen for the online version
- fix the inference, there is still a problem left (clearly). V/J work great (makes sense), but D and insertions fail. Similarly delV fails and delJ mostly fails. There's clearly a problem with delv. Best guesss: there's a problem with the normalisation ?








Current status:
- speed is ok (50 seqs/s roughly ?). Could be slightly faster. I think some range should be replaced by iterator.
- pgen works, but because I consider way more D gene alignment than Quentin some issue when endD and startD are really close to each other.
