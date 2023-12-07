# IHOR

Fast generation:
```{py}
// Create generation model (once only)
gen = ihor.GeneratorVDJ( // ihor.GeneratorVJ
"models/human_T_beta/model_params.txt",
"models/human_T_beta/model_marginals.txt",
"models/human_T_beta/V_gene_CDR3_anchors.csv",
"models/human_T_beta/J_gene_CDR3_anchors.csv")

// Generate productive amino-acid seuqence
result = gen.generate(True)
print(f"Full sequence {result.full_seq}")
print(f"V gene {result.v_name}, J gene {result.j_name}")
print(f"CDR3 {result.cdr3_nt} {result.cdr3_aa}")
```
