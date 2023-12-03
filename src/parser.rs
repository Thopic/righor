use regex::Regex;
use ndarray::{ArrayD, IxDyn};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

// Parser for the marginals and params files

pub struct ParserMarginals {
    marginals: Vec<Marginal>,
}

#[derive(Default)]
struct Marginal {
    name: String,
    dimensions: Vec<usize>, // list of the dimensions
    dependences: Vec<String>, // list of the other variables that Marginal depends on
    probabilities: ArrayD<f64>, // Array that contains all the probabilities of interest
}



impl Marginal {
    pub fn parse(&mut self, str_data: &Vec<String>) -> Result<(), String> {
        if str_data.len() < 2 {
            return Err("Invalid file format.".to_string());
        }

        self.name = str_data[0].trim_start_matches('@').to_string();
        self.dimensions = parse_dim(&str_data[1])?;
        self.dependences = Vec::new();
        let product = self.dimensions.iter().product();
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


impl ParserMarginals {
    pub fn parse(&mut self, filename: &Path) -> Result<(), String>{
	let sections = parse_file(filename)?;
	let m: Marginal = Default::default();
	for s in sections {
	    match s.first() {
		Some(_) =>  m.parse(&s),
		None => Err("Invalid format: empty vector")
	    }
	    self.marginals.push(m);
	}
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
		vec.push(line.trim());
		sections.push(vec);
	    }
	    _ => {
		match sections.last() {
		    Some(v) => v.push(line.trim()),
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
    let re = Regex::new(r"#\[(\w+),(\d+)\]").unwrap();
    let mut texts = Vec::new();
    let mut numbers = Vec::new();

    for cap in re.captures_iter(s) {
        if let (Some(text_match), Some(number_match)) = (cap.get(1), cap.get(2)) {
            let text = text_match.as_str().to_string();
            let number = usize::from_str(number_match.as_str())
                .map_err(|e| format!("Failed to parse number: {}", e))?;
            texts.push(text);
            numbers.push(number);
        } else {
            return Err(format!("Invalid format: {}", s));
        }
    }
    Ok((texts, numbers))
}

fn parse_values(s: &str) -> Result<Vec<f64>, String> {
    s.trim_start_matches('%').split(',')
     .map(|num_str| num_str.parse::<f64>()
                          .map_err(|_| format!("Failed to parse '{}'", num_str)))
     .collect()
}
