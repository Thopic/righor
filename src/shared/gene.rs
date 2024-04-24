//! Deal with V/J gene names and gene representations
use crate::shared::Dna;
use anyhow::{anyhow, Result};
#[cfg(all(feature = "py_binds", feature = "pyo3"))]
use pyo3::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};

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
    pub seq: Dna,
    pub seq_with_pal: Option<Dna>, // Dna with the palindromic insertions (model dependant)
}

#[cfg(all(feature = "py_binds", feature = "pyo3"))]
#[pymethods]
impl Gene {
    fn __repr__(&self) -> String {
        format!("Gene({})", self.name)
    }

    #[new]
    #[pyo3(signature = (name = "", cdr3_pos = None, functional = "", seq = ""))]
    fn new(name: String, cdr3_pos: Option<usize>, functional: String, seq: String) -> Gene {
        Gene {
            name,
            cdr3_pos,
            functional,
            seq,
            seq_with_pal: None,
        }
    }
}

impl Gene {
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

struct GeneNameParser {
    name: String,
    first_index: Option<i32>,
    second_index: Option<i32>,
    _third_index: Option<i32>,
    allele_index: Option<i32>,
    gene: Gene,
}

pub trait ModelGen {
    fn get_v_segments(&self) -> Vec<Gene>;
    fn get_j_segments(&self) -> Vec<Gene>;
}

pub fn genes_matching(x: &str, model: &impl ModelGen, exact: bool) -> Result<Vec<Gene>> {
    let regex = Regex::new(
        r"^(TRB|TRA|IGH|IGK|IGL|TRG|TRD)(?:\w+)?(V|D|J)([\w-]+)?(?:/DV\d+)?(?:\*(\d+))?(?:/OR.*)?$",
    )
    .unwrap();
    let g = regex
        .captures(x)
        .ok_or(anyhow!("Gene {} does not have a valid name", x))?;

    let chain = g.get(1).map_or("", |m| m.as_str());
    let gene_type = g.get(2).map_or("", |m| m.as_str());
    let gene_id = g.get(3).map_or("", |m| m.as_str());
    let allele = g.get(4).and_then(|m| m.as_str().parse::<i32>().ok());

    let possible_genes = igor_genes(chain, gene_type, model)?;

    let gene_id_regex = Regex::new(r"(\d+)(?:[-S](\d+))?").unwrap();
    let gene_id_match = gene_id_regex.captures(gene_id);

    let (gene_id_1, gene_id_2) = gene_id_match.map_or((None, None), |m| {
        let id1 = m.get(1).and_then(|x| x.as_str().parse::<i32>().ok());
        let id2 = m.get(2).and_then(|x| x.as_str().parse::<i32>().ok());
        (id1, id2)
    });

    let result: Vec<Gene> = if exact {
        possible_genes
            .into_iter()
            .filter(|a| a.name == x)
            .map(|x| x.gene)
            .collect()
    } else {
        possible_genes
            .into_iter()
            .filter(|a| match (gene_id_1, gene_id_2, allele) {
                (Some(id1), Some(id2), Some(al)) => {
                    a.first_index == Some(id1)
                        && a.second_index == Some(id2)
                        && a.allele_index == Some(al)
                }
                (Some(id1), Some(id2), None) => {
                    a.first_index == Some(id1) && a.second_index == Some(id2)
                }
                (Some(id1), None, Some(al)) => {
                    a.first_index == Some(id1) && a.allele_index == Some(al)
                }
                (Some(id1), None, None) => a.first_index == Some(id1),
                _ => false,
            })
            .map(|x| x.gene)
            .collect()
    };

    Ok(result)
}

fn igor_genes(chain: &str, gene_type: &str, model: &impl ModelGen) -> Result<Vec<GeneNameParser>> {
    let regex =
        Regex::new(r"(\d+)(?:P)?(?:[\-S](\d+)(?:D)?(?:\-(\d+))?)?(?:/DV\d+)?(?:-NL1)?(?:\*(\d+))?")
            .unwrap();

    let vsegments = model.get_v_segments();
    let jsegments = model.get_j_segments();
    let list_genes = match gene_type {
        "V" => &vsegments,
        "J" => &jsegments,
        _ => return Err(anyhow!("Gene type {} is not valid", gene_type)),
    };

    let key = format!("{}{}", chain, gene_type);
    let mut lst: Vec<GeneNameParser> = Vec::new();

    for gene_obj in list_genes {
        let gene = &gene_obj.name;
        if let Some(cap) = regex.captures(&(key.clone() + gene)) {
            let gene_parser = GeneNameParser {
                name: gene.clone(),
                first_index: cap.get(1).and_then(|m| m.as_str().parse::<i32>().ok()),
                second_index: cap.get(2).and_then(|m| m.as_str().parse::<i32>().ok()),
                _third_index: cap.get(3).and_then(|m| m.as_str().parse::<i32>().ok()),
                allele_index: cap.get(4).and_then(|m| m.as_str().parse::<i32>().ok()),
                gene: gene_obj.clone(),
            };
            lst.push(gene_parser);
        } else {
            return Err(anyhow!("{} does not match. Check if the gene name and the model are compatible (e.g(e.g. TRA for a TRB/IGL model)", key));
        }
    }
    Ok(lst)
}
