// Parser for the marginals and params files
use regex::Regex;
use ndarray::{ArrayD, IxDyn};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::str::FromStr;
use crate::utils::{Gene, Dna};
use std::collections::HashMap;



#[derive(Default, Clone, Debug)]
pub struct ParserMarginals {
    marginals: Vec<Marginal>,
}

#[derive(Default, Clone, Debug)]
struct Marginal {
    name: String,
    dimensions: Vec<usize>, // list of the dimensions
    dependences: Vec<String>, // list of the other variables that Marginal depends on
    probabilities: ArrayD<f64>, // Array that contains all the probabilities of interest
}



// an event is either a gene or a # of insertion/deletion
#[derive(Clone, Debug)]
enum Event {
    Default,
    Gene(Gene),
    Number(i64)
}

impl Default for Event {
    fn default() -> Self {
        Event::Default
    }
}

#[derive(Default, Clone, Debug)]
pub struct ParserParams {
    params: HashMap<String, Vec<Event>>,
    error_rate: f64
}








impl Marginal {
    pub fn parse(&mut self, str_data: &Vec<String>) -> Result<(), String> {
        if str_data.len() < 2 {
            return Err("Invalid file format.".to_string());
        }

        self.name = str_data[0].trim_start_matches('@').to_string();
        self.dimensions = parse_dim(&str_data[1])?;
        self.dependences = Vec::new();
	// remove the last dimension
        let product = self.dimensions.iter().take(self.dimensions.len() - 1).product();
        self.probabilities = ArrayD::zeros(IxDyn(&self.dimensions));

        for ii in 0..product {
            let dependences_line = str_data.get(2 * ii + 2)
                .ok_or(format!("Invalid file format for the marginal {}", self.name))?;
            let (dependences, indexes) = parse_dependence(dependences_line)?;
            match self.dependences.len() {
                0 => self.dependences = dependences,
		_ => if self.dependences != dependences {
		    return Err(format!("Invalid file format for the marginal {}", self.name));
		}
            }

            let values_line = str_data.get(2 * ii + 3)
                .ok_or(format!("Invalid file format for the marginal {}", self.name))?;
            let values = parse_values(values_line)?;

            for (idx, val) in values.iter().enumerate() {
                let mut idxs = indexes.clone();
                idxs.push(idx);
                self.probabilities[IxDyn(&idxs)] = *val;
            }
        }

        Ok(())
    }
}


fn parse_genes(str_data: &Vec<String>) -> Result<Vec<Event>, String> {
    let mut events:Vec<Event> = vec![Default::default(); str_data.len()];
    for line in str_data{
	let data: Vec<String> = line.split(';').map(|s| s.to_string()).collect();
	if data.len() != 3 {
	    return Err(format!("Invalid format for gene event {}", line))
	    }
	else{
	    let gene = Gene {
		name: data[0].split('|').nth(1).ok_or("Wrong format for gene name")?.to_string(),
		functional: data[0].split('|').nth(3).ok_or("Wrong format for gene name")?.to_string(),
		seq: Dna::from_string(&data[1])
		};
	    let index = usize::from_str(&data[2]).map_err(|e| e.to_string())?;
	    events[index] = Event::Gene(gene);
	}
    }
    Ok(events)
}

fn parse_numbers(str_data: &Vec<String>) -> Result<Vec<Event>, String> {
    let mut events:Vec<Event> = vec![Default::default(); str_data.len()];
    for line in str_data{
	let data: Vec<String> = line.split(';').map(|s| s.to_string()).collect();
	    if data.len() != 2 {
		return Err(format!("Invalid format for gene event {}", line))
	    }
	else {
	    let value = i64::from_str(&data[0][1..]).map_err(|e| e.to_string())?;
	    let index = usize::from_str(&data[1]).map_err(|e| e.to_string())?;
	    events[index] = Event::Number(value);
	}
    }
    Ok(events)
}



impl ParserParams {
    pub fn parse(&mut self, filename: &Path) -> Result<(), String>{
	let sections = parse_file(filename)?;
	for s in sections {
	    match s.first() {
		Some(string) => match string.as_str() {
		    "@Event_list" => self.parse_event_list(&s),
		    "@Edges" => {Ok(())},
		    "@ErrorRate" => self.parse_error_rate(&s),
		    _ => Err(format!("Invalid format: wrong key {}", string)),
		}
		None => Err("Invalid format: empty vector".to_string()),
	    }?;
	}
	Ok(())
    }

    fn parse_event(&mut self, str_data: &Vec<String>) -> Result<(), String>{
	if str_data.len() < 2 {
            return Err("Invalid file format.".to_string());
        }
	let name = str_data.get(0).ok_or("Invalid file format")?;
	let key = name.split(';').last().ok_or(format!("Invalid file format, {}", name))?.to_string();
	// list of genes
	if name.starts_with("#GeneChoice") {
	    let genes_str = str_data.iter().skip(1).map(|s| s.clone()).collect();
	    self.params.insert(key, parse_genes(&genes_str)?);
	}
	else if name.starts_with("#Deletion") | name.starts_with("#Insertion") {
	    let number_str = str_data.iter().skip(1).map(|s| s.clone()).collect();
	    self.params.insert(key, parse_numbers(&number_str)?);
	}
	else if name.starts_with("#DinucMarkov") {
	    // Nothing, assume that this wasn't changed (this is stupid anyhow)
	}
	else {
	    return Err("Invalid format, wrong key in the Event_list".to_string());
	}
	Ok(())
    }


    fn parse_error_rate(&mut self, str_data: &Vec<String>) -> Result<(), String>{
	if str_data.len() != 3 {
	    return Err("Invalid format (error rate)".to_string());
	}
	self.error_rate = str_data[2]
	    .parse::<f64>()
	    .map_err(|_| format!("Failed to parse '{}'", str_data[2]))?;
	Ok(())
    }

    fn parse_event_list(&mut self, str_data: &Vec<String>) -> Result<(), String>{
	let mut events: Vec<Vec<String>> = Vec::new();
	for line in str_data.iter().skip(1) {
	    match line.chars().next() {
		Some('#') => {
		    let mut vec = Vec::new();
		    vec.push(line.to_string());
		    events.push(vec);
		}
		_ => {
		    match events.last_mut() {
			Some(ref mut v) => v.push(line.to_string()),
			None => return Err("Invalid file format: error with the first line".to_string())
		    }
		}
	    }
	}
	for ev in events {
	    self.parse_event(&ev);
	}
	Ok(())
    }
}


impl ParserMarginals {
    pub fn parse(&mut self, filename: &Path) -> Result<(), String>{
	let sections = parse_file(filename)?;
	for s in sections {
	    let mut m: Marginal = Default::default();
	    match s.first() {
		Some(_) =>  m.parse(&s),
		None => Err("Invalid format: empty vector".to_string())
	    };
	    self.marginals.push(m);
	}
	Ok(())
    }

}



pub fn parse_file(filename: &Path) -> Result<Vec<Vec<String>>, &'static str> {
    let mut sections: Vec<Vec<String>> = Vec::new();
    let file = File::open(filename).map_err(|_| "Unable to open file")?;
    let reader = io::BufReader::new(file);
    for line_result in reader.lines() {
	let line = line_result.map_err(|_| "Invalid file format: error reading line")?;
	match line.chars().next() {
	    Some('@') => {
		let mut vec = Vec::new();
		vec.push(line.trim().to_string());
		sections.push(vec);
	    }
	    _ => {
		match sections.last_mut() {
		    Some(ref mut v) => v.push(line.trim().to_string()),
		    None => return Err("Invalid file format: error with the first line")
		}
	    }
	}
    }
    Ok(sections)
}




fn parse_dim(s: &str) -> Result<Vec<usize>, String> {
    let re = Regex::new(r"^\$Dim\[(\d+(?:,\d+)*)\]$").unwrap();
    if let Some(caps) = re.captures(s) {
        caps.get(1).unwrap().as_str()
            .split(',')
            .map(|num| usize::from_str(num).map_err(|e| e.to_string()))
            .collect()
    } else {
        Err(format!("Invalid format: {}", s))
    }
}



fn parse_dependence(s: &str) -> Result<(Vec<String>, Vec<usize>), String> {
    // Parse lines like "#[v_choice,0],[j_choice,6]" return ["v_choice", "j_choice"] and [0, 6]
    if s == "#" {
	return Ok((Vec::new(), Vec::new()));
    }
    let re = Regex::new(r"^\[(?<text>\w+),(?<number>\d+)\]$").unwrap();
    let mut texts = Vec::new();
    let mut numbers = Vec::new();

    for dep_str in s.chars().skip(1).collect::<String>().split(',') {
	let Some(caps) = re.captures(s) else {return Err(format!("Invalid format: {}", s))};
	let text = caps["text"].to_string();
        let number = usize::from_str(&caps["number"])
                .map_err(|e| format!("Failed to parse number: {}", e))?;
        texts.push(text);
        numbers.push(number);
    }
    Ok((texts, numbers))
}

fn parse_values(s: &str) -> Result<Vec<f64>, String> {
    s.trim_start_matches('%').split(',')
     .map(|num_str| num_str.parse::<f64>()
                          .map_err(|_| format!("Failed to parse '{}'", num_str)))
     .collect()
}
