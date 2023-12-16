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
