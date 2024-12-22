import righor
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from tqdm.notebook import tqdm
from collections import Counter
import numpy as np

# useful if there's a weird error
# import os
# os.environ["RUST_BACKTRACE"] = "1"

# load the model
igor_model = righor.load_model("human", "trb")
# alternatively, you can load a model from igor files
# igor_model = righor.load_model_from_files(params.txt, marginals.txt, anchor_v.csv, anchor_j.csv)

# constant error: all sequences have the same number of errors, here no errors
igor_model.error = righor.ErrorParameters.constant_error(0.0)
# uniform error: the number of error in the sequence is sequence-specific
# and inferred using the V gene (useful for sequences that undergo somatic hypermutation)
igor_model.error = righor.ErrorParameters.uniform_error()


# Inference of a model
# to keep this short we are artifically reducing the number of genes
igor_model.v_segments = [igor_model.v_segments[0], igor_model.v_segments[1]]
igor_model.j_segments = [igor_model.j_segments[0], igor_model.j_segments[1]]
# a good inference with all genes would take ≈ 10'000 sequences and a few hours on a laptop
# (don't do it on a laptop)


# here we just generate the sequences needed
generator = igor_model.generator()
example_seq = generator.generate(False)
sequences = [generator.generate(False).full_seq for _ in range(1000)]

align_params = righor.AlignmentParameters()
infer_params = righor.InferenceParameters()
# define parameters for the alignment and the inference
# (can also be done for the evaluation)
# the default are generally fine
# longer V gene cut-off improve the alignment but slower
align_params.left_v_cutoff = 90
# define the likelihood cutoff (higher: less precise but faster)
infer_params.min_ratio_likelihood = 1e-4
# infer_params.min_likelihood = 1e-20

# generate an uniform model as a starting point
# (it's generally much better to use an already inferred model as the starting point)
model = igor_model.copy().uniform()

# multiple round of expectation-maximization to infer the model
models = {}
model = igor_model.uniform()
models[0] = model
for ii in range(20):
    models[ii+1] = models[ii].copy()
    models[ii+1].infer(sequences,
                       align_params,
                       infer_params)
