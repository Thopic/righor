// Parser for the marginals and params files

use crate::sequence::Dna;
use crate::shared::utils::Gene;
use anyhow::{anyhow, Result};
use csv::Reader;
use ndarray::{ArrayD, IxDyn};
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, Read};
use std::path::Path;
use std::str::FromStr;

#[derive(Default, Clone, Debug)]
pub struct ParserMarginals {
    pub marginals: HashMap<String, Marginal>,
}

#[derive(Default, Clone, Debug)]
pub struct Marginal {
    pub dimensions: Vec<usize>,     // list of the dimensions
    pub dependences: Vec<String>,   // list of the other variables that Marginal depends on
    pub probabilities: ArrayD<f64>, // Array that contains all the probabilities of interest
}

// an event is either a gene or a # of insertion/deletion
#[derive(Clone, Debug)]
pub enum EventType {
    Genes(Vec<Gene>),
    Numbers(Vec<i64>),
}

impl EventType {
    pub fn write(&self) -> String {
        let mut result = String::new();
        match self {
            EventType::Genes(v) => {
                for (ii, g) in v.iter().enumerate() {
                    result.push_str(&format!("%{};{};{}\n", g.name, g.seq, ii));
                }
            }
            EventType::Numbers(v) => {
                for (ii, nb) in v.iter().enumerate() {
                    result.push_str(&format!("%{};{}\n", nb, ii));
                }
            }
        }
        result
    }

    pub fn to_genes(&self) -> Result<Vec<Gene>> {
        match self {
            EventType::Genes(v) => Ok(v.to_vec()),
            _ => Err(anyhow!("Wrong conversion for the EventType (not genes)"))?,
        }
    }

    pub fn to_numbers(&self) -> Result<Vec<i64>> {
        match self {
            EventType::Numbers(v) => Ok(v.to_vec()),
            _ => Err(anyhow!("Wrong conversion for the EventType (not numbers)"))?,
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct ParserParams {
    pub params: HashMap<String, EventType>,
    pub error_rate: f64,
}

impl Marginal {
    pub fn create(dependences: Vec<&str>, probabilities: ArrayD<f64>) -> Marginal {
        let dimensions: Vec<usize> = probabilities
            .shape()
            .iter()
            .map(|&d| d.try_into().unwrap())
            .collect();
        Marginal {
            dimensions,
            dependences: dependences.into_iter().map(String::from).collect(),
            probabilities,
        }
    }

    pub fn write(&self) -> Result<String> {
        let mut result = format!(
            "$Dim[{}]\n",
            self.dimensions
                .iter()
                .map(usize::to_string)
                .collect::<Vec<String>>()
                .join(",")
        );

        if (self.dependences.len() + 1) != self.dimensions.len()
            || self.dimensions.len() != self.probabilities.shape().len()
        {
            return Err(anyhow!("Corrupted marginal struct."));
        }

        for (i, &dim_size) in self.dimensions.iter().enumerate() {
            if self.probabilities.shape()[i] != dim_size {
                return Err(anyhow!("Corrupted marginal struct."));
            }
        }

        if self.dimensions.len() == 1 {
            result.push_str(&"#\n%");
            for k in 0..self.dimensions[0] {
                let prob = self.probabilities[k];
                result.push_str(&format!("{},", prob));
            }
            result.pop(); // remove last comma
            result.push('\n');
        } else if self.dimensions.len() == 2 {
            for i in 0..self.dimensions[0] {
                result.push_str(&format!("#[{},{}]\n", self.dependences[0], i));
                for k in 0..self.dimensions[1] {
                    let prob = self.probabilities[[i, k]];
                    result.push_str(&format!("{},", prob));
                }
                result.pop(); // remove last comma
                result.push('\n');
            }
        } else if self.dimensions.len() == 3 {
            for i in 0..self.dimensions[0] {
                for j in 0..self.dimensions[1] {
                    result.push_str(&format!(
                        "#[{},{}],[{},{}]\n",
                        self.dependences[0], i, self.dependences[1], j,
                    ));

                    for k in 0..self.dimensions[2] {
                        let prob = self.probabilities[[i, j, k]];
                        result.push_str(&format!("{},", prob));
                    }
                    result.pop(); // Remove the last comma
                    result.push('\n');
                }
            }
        } else {
            return Err(anyhow!("Not implemented."));
        }
        Ok(result)
    }

    pub fn parse(str_data: &Vec<String>) -> Result<(String, Marginal)> {
        if str_data.len() < 2 {
            return Err(anyhow!("Invalid file format."))?;
        }
        let key = str_data[0].trim_start_matches('@').to_string();
        let dimensions = parse_dim(&str_data[1])?;
        let mut marg = Marginal {
            dimensions,
            ..Default::default()
        };
        // remove the last dimension
        let product = marg
            .dimensions
            .iter()
            .take(marg.dimensions.len() - 1)
            .product();
        marg.probabilities = ArrayD::zeros(IxDyn(&marg.dimensions));

        for ii in 0..product {
            let dependences_line = str_data.get(2 * ii + 2).ok_or(anyhow!(format!(
                "Invalid file format for the marginal {}",
                key
            )));
            let (dependences, indexes) = parse_dependence(dependences_line?)?;
            match marg.dependences.len() {
                0 => marg.dependences = dependences,
                _ => {
                    if marg.dependences != dependences {
                        return Err(anyhow!(format!(
                            "Invalid file format for the marginal {}",
                            key
                        )))?;
                    }
                }
            }

            let values_line = str_data.get(2 * ii + 3).ok_or(anyhow!(format!(
                "Invalid file format for the marginal {}",
                key
            )))?;
            let values = parse_values(values_line)?;
            for (idx, val) in values.iter().enumerate() {
                let mut idxs = indexes.clone();
                idxs.push(idx);
                marg.probabilities[IxDyn(&idxs)] = *val;
            }
        }

        Ok((key, marg))
    }
}

fn parse_genes(str_data: &Vec<String>) -> Result<EventType> {
    let mut events: Vec<Gene> = vec![Default::default(); str_data.len()];
    for line in str_data {
        let data: Vec<String> = line.split(';').map(|s| s.to_string()).collect();
        if data.len() != 3 {
            Err(anyhow!(format!("Invalid format for gene event {}", line)))?;
        } else {
            let gene = Gene {
                name: data[0].chars().skip(1).collect(),
                functional: "".to_string(), // not available from this file
                seq: Dna::from_string(&data[1])?,
                seq_with_pal: None,
                cdr3_pos: Some(0), // not available from this file
            };
            let index = usize::from_str(&data[2])?;
            events[index] = gene;
        }
    }
    Ok(EventType::Genes(events))
}

fn parse_numbers(str_data: &Vec<String>) -> Result<EventType> {
    let mut events: Vec<i64> = vec![0; str_data.len()];
    for line in str_data {
        let data: Vec<String> = line.split(';').map(|s| s.to_string()).collect();
        if data.len() != 2 {
            Err(anyhow!(format!("Invalid format for gene event {}", line)))?;
        } else {
            let value = i64::from_str(&data[0][1..].trim())?;
            let index = usize::from_str(&data[1].trim())?;
            events[index] = value;
        }
    }
    Ok(EventType::Numbers(events))
}

impl ParserParams {
    pub fn parse(sections: Vec<Vec<String>>) -> Result<ParserParams> {
        let mut pp: ParserParams = Default::default();
        for s in sections {
            match s.first() {
                Some(string) => match string.as_str() {
                    "@Event_list" => pp.parse_event_list(&s),
                    "@Edges" => Ok(()),
                    "@ErrorRate" => pp.parse_error_rate(&s),
                    _ => Err(anyhow!(format!("Invalid format: wrong key {}", string)))?,
                },
                None => Err(anyhow!("Invalid format: empty vector".to_string()))?,
            }?;
        }
        Ok(pp)
    }

    pub fn add_anchors_gene<R: Read>(&mut self, reader: R, gene_choice: &str) -> Result<()> {
        let mut rdr = Reader::from_reader(reader);
        let mut anchors = HashMap::<String, usize>::new();
        let mut functions = HashMap::<String, String>::new();
        rdr.headers()
            .map_err(|_e| anyhow!("Error reading the anchor file headers"))?;
        // TODO: check that the headers are right
        for result in rdr.records() {
            let record = result.map_err(|e| anyhow!("Error reading the record {:?}", e))?;
            let gene_name = record.get(0).unwrap();
            anchors.insert(
                gene_name.to_string(),
                usize::from_str(record.get(1).unwrap())
                    .map_err(|e| anyhow!("Error reading the anchor file headers: {:?}", e))?,
            );
            functions.insert(gene_name.to_string(), record.get(2).unwrap().to_string());
        }

        if let Some(EventType::Genes(v)) = self.params.get_mut(gene_choice) {
            v.iter_mut().for_each(|g| {
                g.cdr3_pos = Some(
                    *anchors
                        .get(&g.name)
                        .ok_or(anyhow!("{} not found in anchor file", g.name))
                        .unwrap(),
                );
                g.functional = functions
                    .get(&g.name)
                    .ok_or(anyhow!("{} not found in anchor file", g.name))
                    .unwrap()
                    .clone()
            });
        } else {
            return Err(anyhow!("Wrong value for gene_choice (add_anchors_gene)"))?;
        }
        Ok(())
    }

    fn parse_event(&mut self, str_data: &Vec<String>) -> Result<()> {
        if str_data.len() < 2 {
            return Err(anyhow!("Invalid file format."))?;
        }
        let name = str_data.get(0).ok_or(anyhow!("Invalid file format"))?;
        let key = name
            .split(';')
            .last()
            .ok_or(anyhow!(format!("Invalid file format, {}", name)))?
            .to_string();
        // list of genes
        if name.starts_with("#GeneChoice") {
            let genes_str = str_data.iter().skip(1).cloned().collect();
            self.params.insert(key, parse_genes(&genes_str)?);
        } else if name.starts_with("#Deletion") | name.starts_with("#Insertion") {
            let number_str = str_data.iter().skip(1).cloned().collect();
            self.params.insert(key, parse_numbers(&number_str)?);
        } else if name.starts_with("#DinucMarkov") {
            // Nothing, assume that this wasn't changed
        } else {
            return Err(anyhow!("Invalid format, wrong key in the Event_list"))?;
        }
        Ok(())
    }

    fn parse_error_rate(&mut self, str_data: &Vec<String>) -> Result<()> {
        if str_data.len() != 3 {
            return Err(anyhow!("Invalid format (error rate)"))?;
        }
        self.error_rate = str_data[2]
            .parse::<f64>()
            .map_err(|_| anyhow!(format!("Failed to parse '{}'", str_data[2])))?;
        Ok(())
    }

    fn parse_event_list(&mut self, str_data: &[String]) -> Result<()> {
        let mut events: Vec<Vec<String>> = Vec::new();
        for line in str_data.iter().skip(1) {
            match line.chars().next() {
                Some('#') => {
                    let vec = vec![line.to_string()];
                    events.push(vec);
                }
                _ => match events.last_mut() {
                    Some(ref mut v) => v.push(line.to_string()),
                    None => return Err(anyhow!("Invalid file format: error with the first line"))?,
                },
            }
        }
        for ev in events {
            self.parse_event(&ev)?;
        }
        Ok(())
    }
}

impl ParserMarginals {
    pub fn parse(sections: Vec<Vec<String>>) -> Result<ParserMarginals> {
        let mut pm: ParserMarginals = Default::default();
        for s in sections {
            match s.first() {
                Some(_) => {
                    let (key, marg) = Marginal::parse(&s)?;
                    pm.marginals.insert(key, marg);
                }
                None => return Err(anyhow!("Invalid format: empty vector"))?,
            };
        }
        Ok(pm)
    }
}

pub fn parse_file(filename: &Path) -> Result<Vec<Vec<String>>> {
    let mut sections: Vec<Vec<String>> = Vec::new();
    let file = File::open(filename).map_err(|_| anyhow!("Unable to open file"))?;
    let reader = io::BufReader::new(file);
    for line_result in reader.lines() {
        let line = line_result.map_err(|_| anyhow!("Invalid file format: error reading line"))?;
        match line.chars().next() {
            Some('@') => {
                let vec = vec![line.trim().to_string()];
                sections.push(vec);
            }
            _ => match sections.last_mut() {
                Some(ref mut v) => v.push(line.trim().to_string()),
                None => return Err(anyhow!("Invalid file format: error with the first line")),
            },
        }
    }
    Ok(sections)
}

pub fn parse_str(data: &str) -> Result<Vec<Vec<String>>> {
    let mut sections: Vec<Vec<String>> = Vec::new();
    for line in data.lines() {
        match line.chars().next() {
            Some('@') => {
                let vec = vec![line.trim().to_string()];
                sections.push(vec);
            }
            _ => match sections.last_mut() {
                Some(ref mut v) => v.push(line.trim().to_string()),
                None => return Err(anyhow!("Invalid file format: error with the first line")),
            },
        }
    }
    Ok(sections)
}

fn parse_dim(s: &str) -> Result<Vec<usize>> {
    let re = Regex::new(r"^\$Dim\[(\d+(?:,\d+)*)\]$").unwrap();
    if let Some(caps) = re.captures(s) {
        Ok(caps
            .get(1)
            .unwrap()
            .as_str()
            .split(',')
            .map(|num_str| num_str.trim()) // Trim whitespace around the numbers
            .filter_map(|num| usize::from_str(num).ok())
            .collect())
    } else {
        Err(anyhow!(format!("Invalid format: {}", s)))?
    }
}

fn parse_dependence(s: &str) -> Result<(Vec<String>, Vec<usize>)> {
    // Parse lines like "#[v_choice,0],[j_choice,6]" return ["v_choice", "j_choice"] and [0, 6]
    if s == "#" {
        return Ok((Vec::new(), Vec::new()));
    }
    let re = Regex::new(r"^(?<text>\w+),\s*(?<number>\d+)\]").unwrap();
    let mut texts = Vec::new();
    let mut numbers = Vec::new();

    for dep_str in s.split('[').skip(1) {
        let caps = re
            .captures(dep_str)
            .ok_or(anyhow!(format!("Invalid format: {}", s)))?;
        let text = caps["text"].to_string();
        let number = usize::from_str(&caps["number"])?;
        texts.push(text);
        numbers.push(number);
    }
    Ok((texts, numbers))
}

fn parse_values(s: &str) -> Result<Vec<f64>> {
    Ok(s.trim_start_matches('%')
        .split(',')
        .map(|num_str| num_str.trim()) // Trim whitespace around the numbers
        .filter_map(|num_str| num_str.parse::<f64>().ok())
        .collect())
}
