// Parser for the marginals and params files



struct Marginal {
    name: String,
    dimension: Vec<usize>,
    values: Vec<f64>,
}


pub struct Parser{
    marginals: Vec<Marginal>,




}
