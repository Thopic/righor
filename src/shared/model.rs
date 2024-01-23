use crate::sequence::Dna;
use crate::shared::Gene;
use anyhow::{anyhow, Result};

pub fn sanitize_v(genes: Vec<Gene>) -> Result<Vec<Dna>> {
    // Add palindromic inserted nucleotides to germline V sequences and cut all
    // sequences to only keep their CDR3 parts
    let mut cut_genes = Vec::<Dna>::new();
    for g in genes {
        // some V-genes are not complete. They don't appear in the model, but we
        // can't ignore them
        // TODO: I need to change the way this is done...
        if g.cdr3_pos.unwrap() >= g.seq.len() {
            cut_genes.push(Dna::new());
            continue;
        }

        let gene_seq: Dna = g
            .seq_with_pal
            .ok_or(anyhow!("Palindromic sequences not created"))?;

        let cut_gene: Dna = Dna {
            seq: gene_seq.seq[g.cdr3_pos.unwrap()..].to_vec(),
        };
        cut_genes.push(cut_gene);
    }
    Ok(cut_genes)
}

pub fn sanitize_j(genes: Vec<Gene>, max_del_j: usize) -> Result<Vec<Dna>> {
    // Add palindromic inserted nucleotides to germline J sequences and cut all
    // sequences to only keep their CDR3 parts
    let mut cut_genes = Vec::<Dna>::new();
    for g in genes {
        let gene_seq: Dna = g
            .seq_with_pal
            .ok_or(anyhow!("Palindromic sequences not created"))?;

        // for J, we want to also add the last CDR3 amino-acid (F/W)
        let cut_gene: Dna = Dna {
            seq: gene_seq.seq[..g.cdr3_pos.unwrap() + 3 + max_del_j].to_vec(),
        };
        cut_genes.push(cut_gene);
    }
    Ok(cut_genes)
}
