use crate::shared::{AlignmentParameters, InferenceParameters};
use crate::shared::{Dna, Gene};
use crate::vdj::{ResultInference, Sequence};
use anyhow::{anyhow, Result};
use rand::Rng;
use std::path::Path;

/// Generic trait to include all the models
pub trait Modelable {
    type GenerationResult;
    type RecombinaisonEvent;

    /// Load the model by looking its name in a database
    fn load_from_name(
        species: &str,
        chain: &str,
        id: Option<String>,
        model_dir: &Path,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Load the model from a set of files in IGoR format
    fn load_from_files(
        path_params: &Path,
        path_marginals: &Path,
        path_anchor_vgene: &Path,
        path_anchor_jgene: &Path,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Load the model from a set of String in IGoR format
    fn load_from_str(
        params: &str,
        marginals: &str,
        anchor_vgene: &str,
        anchor_jgene: &str,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Save the model in a given directory (write 4 files)
    fn save_model(&self, directory: &Path) -> Result<()>;

    /// Save the data in json format
    fn save_json(&self, filename: &Path) -> Result<()>
    where
        Self: Sized;

    /// Load a model saved in json format
    fn load_json(filename: &Path) -> Result<Self>
    where
        Self: Sized;

    /// Update the internal state of the model so it stays consistent
    fn initialize(&mut self) -> Result<()>;

    /// Generate a sequence
    fn generate<R: Rng>(&mut self, functional: bool, rng: &mut R) -> Self::GenerationResult;

    /// Generate a sequence without taking into account the error rate
    fn generate_without_errors<R: Rng>(
        &mut self,
        functional: bool,
        rng: &mut R,
    ) -> Self::GenerationResult;

    /// Return an uniform model (for initializing the inference)
    fn uniform(&self) -> Result<Self>
    where
        Self: Sized;

    /// Evaluate the sequence and return a `ResultInference` object
    fn evaluate(
        &self,
        sequence: &Sequence,
        inference_params: &InferenceParameters,
    ) -> Result<ResultInference>;

    /// Run one round of expectation-maximization on the current model and return the next model.
    fn infer(
        &mut self,
        sequences: &Vec<Sequence>,
        inference_params: &InferenceParameters,
    ) -> Result<()>;

    /// Given a cdr3 sequence + V/J genes return a "aligned" `Sequence` object
    fn align_from_cdr3(
        &self,
        cdr3_seq: Dna,
        vgenes: Vec<Gene>,
        jgenes: Vec<Gene>,
    ) -> Result<Sequence>;

    /// Align one nucleotide sequence and return a `Sequence` object
    fn align_sequence(&self, dna_seq: Dna, align_params: &AlignmentParameters) -> Result<Sequence>;

    /// Recreate the full sequence from the CDR3/vgene/jgene
    fn recreate_full_sequence(&self, dna_cdr3: &Dna, vgene: &Gene, jgene: &Gene) -> Dna;

    /// Test if self is similar to another model
    fn similar_to(&self, m: Self) -> bool;

    /// Return the same model, with only a subset of v genes kept
    fn filter_vs(&self, vs: Vec<Gene>) -> Result<Self>
    where
        Self: Sized;
    /// Return the same model, with only a subset of j genes kept
    fn filter_js(&self, vs: Vec<Gene>) -> Result<Self>
    where
        Self: Sized;
}

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
