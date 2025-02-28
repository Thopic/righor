//! Deal with V/J gene names and gene representations
use crate::shared::sequence::Dna;
use anyhow::{anyhow, Result};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Define some storage wrapper for the V/D/J genes
#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gene {
    pub name: String,
    // start (for V gene) or end (for J gene) of CDR3
    // for V gene this corresponds to the position of the first nucleotide of the "C"
    // for J gene this corresponds to the position of the first nucleotide of the "F/W"
    pub cdr3_pos: Option<usize>,
    pub functional: String,
    pub is_functional: bool,
    pub seq: Dna,
    pub seq_with_pal: Option<Dna>, // Dna with the palindromic insertions (model dependant)

    // helpful to identify gene of the same type
    pub imgt: ImgtRepresentation,
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl Gene {
    fn __repr__(&self) -> String {
        format!("Gene({})", self.name)
    }
    #[new]
    #[pyo3(signature = (name = String::new(), cdr3_pos = None, functional = String::new(), seq = Dna::new()))]
    pub fn py_new(
        name: String,
        cdr3_pos: Option<usize>,
        functional: String,
        seq: Dna,
    ) -> Result<Gene> {
        Gene::new(name, cdr3_pos, functional, seq)
    }
}

impl Gene {
    pub fn new(
        name: String,
        cdr3_pos: Option<usize>,
        functional: String,
        seq: Dna,
    ) -> Result<Gene> {
        let imgt = get_imgt_representation(&name).map_err(|_| anyhow!("Gene names must follow IMGT conventions, e.g. TRAV2-2*03 (error coming from the name {})", name))?;
        Ok(Gene {
            name,
            cdr3_pos,
            functional: functional.clone(),
            is_functional: (functional == "F" || functional == "(F)"),
            seq,
            seq_with_pal: None,
            imgt,
        })
    }

    pub fn is_functional(&self) -> bool {
        self.is_functional
    }

    pub fn set_functional(&mut self, func_str: String) {
        self.functional = func_str;
        self.is_functional = self.functional == "F" || self.functional == "(F)";
    }

    pub fn create_palindromic_ends(&mut self, lenleft: usize, lenright: usize) {
        let palindromic_extension_left = self
            .seq
            .extract_subsequence(0, lenleft)
            .reverse_complement();
        let mut seqpal: Vec<u8> = palindromic_extension_left
            .seq
            .into_iter()
            .chain(self.seq.seq.clone())
            .collect();
        let palindromic_extension_right = self
            .seq
            .extract_subsequence(self.seq.len() - lenright, self.seq.len())
            .reverse_complement();
        seqpal.extend(palindromic_extension_right.seq);

        self.seq_with_pal = Some(Dna {
            seq: seqpal.clone(),
        });
    }
}

pub trait ModelGen {
    fn get_v_segments(&self) -> Vec<Gene>;
    fn get_j_segments(&self) -> Vec<Gene>;
    fn get_d_segments(&self) -> Result<Vec<Gene>>;

    fn genes_matching(&self, x: &str, exact: bool) -> Result<Vec<Gene>>
    where
        Self: Sized,
    {
        let imgt = get_imgt_representation(x).map_err(|_|
						      anyhow!("Gene names must follow IMGT-like conventions, e.g. TRAV2-2*03 (error coming from the name {})", x))?;

        let possible_genes = match imgt.gene_type.as_str() {
            "V" => Ok(self.get_v_segments()),
            "J" => Ok(self.get_j_segments()),
            "D" => self
                .get_d_segments()
                .map_err(|_| anyhow!("D gene asked for, but the model is not VDJ.")),
            _ => Err(anyhow!(
                "Invalid gene type in gene name {} (only V,D,J are allowed)",
                x
            )),
        }?;

        let result: Vec<Gene> = if exact {
            possible_genes
                .into_iter()
                .filter(|a| a.name == x)
                .map(|x| x)
                .collect()
        } else {
            possible_genes
                .into_iter()
                .filter(|a| {
                    match (
                        imgt.gene_id.clone(),
                        imgt.gene_position.clone(),
                        imgt.allele_index,
                    ) {
                        (Some(id1), Some(id2), Some(al)) => {
                            a.imgt.gene_id == Some(id1)
                                && a.imgt.gene_position == Some(id2)
                                && a.imgt.allele_index == Some(al)
                        }
                        (Some(id1), Some(id2), None) => {
                            a.imgt.gene_id == Some(id1) && a.imgt.gene_position == Some(id2)
                        }
                        (Some(id1), None, None) => a.imgt.gene_id == Some(id1),
                        _ => false,
                    }
                })
                .collect()
        };

        Ok(result)
    }
}

#[cfg_attr(all(feature = "py_binds", feature = "pyo3"), pyclass(get_all, set_all))]
#[derive(Default, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImgtRepresentation {
    // chain (TRA) and gene type (V) are mandatory
    chain: String,
    gene_type: String,
    // This corresponds to the "family" of the V gene
    // Examples "2D" (duplicated locus), "3/OR2" (genes orphons, isolated),
    // or "V14/DV14" for a TRA gene that can be recombined with delta and alpha.
    gene_id: Option<String>,
    // Map the position of the gene, just a number initially,
    // but can also be 1-1 if there was insertions.
    // or 1D/1N for a duplicated gene
    gene_position: Option<String>,
    // two figure number
    allele_index: Option<i32>,
    family: Option<i32>,
}

pub fn get_imgt_representation(name: &str) -> Result<ImgtRepresentation> {
    let regex = Regex::new(
        r"^(TCRB|TCRA|TCRG|TCRD|TRB|TRA|IGH|IGK|IGL|TRG|TRD)(V|D|J)([\w/]+)?(:?-([\w/-]*))?(?:\*(\d*))?",
    )
    .unwrap();
    let g = regex
        .captures(name)
        .ok_or(anyhow!("Gene {} does not have a valid name", name))?;

    // deal with the possibly weird convention for TCR names
    let chain_map = HashMap::from([
        ("TCRB".to_string(), "TRB".to_string()),
        ("TCRA".to_string(), "TRA".to_string()),
        ("TCRG".to_string(), "TRG".to_string()),
        ("TCRD".to_string(), "TRD".to_string()),
        ("TRB".to_string(), "TRB".to_string()),
        ("TRA".to_string(), "TRA".to_string()),
        ("IGH".to_string(), "IGH".to_string()),
        ("IGK".to_string(), "IGK".to_string()),
        ("IGL".to_string(), "IGL".to_string()),
        ("TRG".to_string(), "TRG".to_string()),
        ("TRD".to_string(), "TRD".to_string()),
    ]);

    let chain = chain_map.get(g.get(1).map_or("", |m| m.as_str())).unwrap();
    let gene_type = g.get(2).map_or("".to_string(), |m| m.as_str().to_string());
    let gene_id = g.get(3).map_or(None, |m| Some(m.as_str().to_string()));
    let gene_position = g.get(4).map_or(None, |m| Some(m.as_str().to_string()));
    let allele_index = g.get(5).and_then(|m| m.as_str().parse::<i32>().ok());

    let family = if gene_id.is_none() {
        None
    } else {
        let gene_family_regex = Regex::new(r"^(\d+)[DN]?(?:/OR\d*)$").unwrap();
        let gene_id_unwrap = gene_id.clone().unwrap();
        let gene_family_match = gene_family_regex.captures(&gene_id_unwrap);
        gene_family_match.map_or(None, |m| {
            m.get(1).and_then(|x| x.as_str().parse::<i32>().ok())
        })
    };

    Ok(ImgtRepresentation {
        chain: chain.to_string(),
        gene_type,
        gene_id,
        gene_position,
        allele_index,
        family,
    })
}
