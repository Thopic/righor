use crate::shared::gene::Gene;
use crate::shared::model::Modelable;
use crate::shared::parameters::AlignmentParameters;
use crate::shared::sequence::{DnaLike, Sequence};
use crate::vdj::model::Model as ModelVDJ;
use anyhow::Result;

#[derive(Clone, Debug)]
/// Make infer a generic function by allowing different entry
pub enum EntrySequence {
    Aligned(Sequence),
    NucleotideSequence(DnaLike),
    NucleotideCDR3((DnaLike, Vec<Gene>, Vec<Gene>)),
}

impl EntrySequence {
    pub fn align(&self, model: &ModelVDJ, align_params: &AlignmentParameters) -> Result<Sequence> {
        match self {
            EntrySequence::Aligned(x) => Ok(x.clone()),
            EntrySequence::NucleotideSequence(seq) => {
                model.align_sequence(seq.clone(), align_params)
            }
            EntrySequence::NucleotideCDR3((seq, v, j)) => model.align_from_cdr3(seq, v, j),
        }
    }

    pub fn compatible_with_inference(&self) -> bool {
        match self {
            EntrySequence::Aligned(s) => !s.sequence.is_ambiguous(),
            EntrySequence::NucleotideSequence(s) => !s.is_ambiguous(),
            EntrySequence::NucleotideCDR3((s, _, _)) => !s.is_ambiguous(),
        }
    }

    pub fn is_protein(&self) -> bool {
        match self {
            EntrySequence::Aligned(s) => s.sequence.is_protein(),
            EntrySequence::NucleotideSequence(s) => s.is_protein(),
            EntrySequence::NucleotideCDR3((s, _, _)) => s.is_protein(),
        }
    }
}
