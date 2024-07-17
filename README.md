# RIGHOR

This package, based on IGoR, is meant to learn models of V(D)J recombination.

It can:
- generate sequences
- evaluate sequences (infer the most likely recombination scenarios)
- compute "pgen"

It's probably easier to use the [companion python package](https://pypi.org/project/righor/) (`pip install righor`), but working in Rust directly should also be viable.


How to use the python package:
------------------------------

Load a model:
```py
import righor
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from tqdm.notebook import tqdm
from collections import Counter
import numpy as np


igor_model = righor.load_model("human", "trb")

# alternatively, you can load a model from igor files
# igor_model = righor.load_model_from_files(params.txt, marginals.txt, anchor_v.csv, anchor_j.csv)
```

Generate sequences fast:

```py
# Create a generator object
generator = igor_model.generator(seed=42) # or igor_model.generator() to run it without a seed

# Generate 10'000 functional sequences (not out-of-frame, no stop codons, right boundaries)
for _ in tqdm(range(10000)):
    # generate_without_errors ignore Igor error model, use "generate" if this is needed
    sequence = generator.generate_without_errors(functional=True)
    if "IGH" in sequence.cdr3_aa:
        print("TRB CDR3 containing \"IGH\":", sequence.cdr3_aa)

# Generate one sequence with a particular V/J genes family
V_genes = righor.genes_matching("TRBV5", igor_model) # return all the V genes that match TRBV5
J_genes = righor.genes_matching("TRBJ", igor_model) # all the J genes
generator = igor_model.generator(seed=42, available_v=V_genes, available_j=J_genes)
generation_result = generator.generate_without_errors(functional=True)
print("Result:")
print(generation_result)
print("Explicit recombination event:")
print(generation_result.recombination_event)
```

Evaluate a given sequence:

```py
## Evaluate a given sequence

my_sequence = "ACCCTCCAGTCTGCCAGGCCCTCACATACCTCTCAGTACCTCTGTGCCAGCAGTGAGGACAGGGACGTCACTGAAGCTTTCTTTGGACAAGGCACC"

# evaluate the sequence
result_inference = igor_model.evaluate(my_sequence)

# Most likely scenario
best_event = result_inference.best_event

print(f"Probability that this specific event chain created the sequence: {best_event.likelihood / result_inference.likelihood:.2f}.")
print(f"Reconstructed sequence (without errors):", best_event.reconstructed_sequence)
print(f"Pgen: {result_inference.pgen:.1e}")
```

Infer a model:

```py
# Inference of a model 
# use a very small number of sequences to keep short (takes ~30s)

# here we just generate the sequences needed
generator = igor_model.generator()
example_seq = generator.generate(False)
sequences = [generator.generate(False).full_seq for _ in range(500)]

# define parameters for the alignment and the inference (also possible for the evaluation)
align_params = righor.AlignmentParameters()
align_params.left_v_cutoff = 70
infer_params = righor.InferenceParameters()

# generate an uniform model as a starting point 
# (it's generally *much* faster to start from an already inferred model)
model = igor_model.copy()
model.p_ins_vd = np.ones(model.p_ins_vd.shape)
model.error_rate = 0

# align multiple sequences at once
aligned_sequences = model.align_all_sequences(sequences, align_params)

# multiple round of expectation-maximization to infer the model
models = {}
model = igor_model.uniform()
model.error_rate = 0
models[0] = model
for ii in tqdm(range(35)):
    models[ii+1] = models[ii].copy()
    models[ii+1].infer(aligned_sequences, infer_params)
```

Visualize and save the model
```py
# visualisation of the results
fig = righor.plot_vdj(*[models[ii] for ii in [10, 2, 1, 0]] + [igor_model],
            plots_kws=[{'label':f'Round #{ii}', 'alpha':0.8} for ii in [10,2, 1, 0]] + [{'label':f'og'}] )
# save the model in the Igor format
# will return an error if the directory already exists
models[10].save_model('test_save')
# load the model
igor_model = righor.vdj.Model.load_model_from_files('test_save/model_params.txt',
                                          'test_save/model_marginals.txt',
                                          'test_save/V_gene_CDR3_anchors.csv',
                                          'test_save/J_gene_CDR3_anchors.csv')

# save the model in json format (one file)
models[10].save_json('test_save.json')
# load the model in json
igor_model = righor.vdj.Model.load_json('test_save.json')
```
Extra stuff:
------------

Main differences with IGoR:
- "dynamic programming" method, instead of summing over all events we first pre-compute over sum of events. This means that we can run it with undefined nucleotides like N (at least in theory, I need to add full support for these).
- The D gene alignment is less constrained

Limitations:
- Need to get rid of any primers/ends on the V gene side before running it
- The reads need to be long enough to fully cover the CDR3 (even when it's particularly long)


Programming stuff:
- There's a wasm version for web use.
- python version is in a different crate now.
- to add a model permanently, add it to "models.json". First model in a category is the default model. Each field is one independant model. The elements in chain and species should always be lower-case.

Current status:
- speed is ok (50 seqs/s roughly ?). Could be slightly faster. I think some range should be replaced by iterator.
- pgen works, but because I consider way more D gene alignment than Quentin some issue when endD and startD are really close to each other.
