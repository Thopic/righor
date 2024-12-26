## Things to do:

- test the inference in detail
- add more tests
- Potential insertion in V/J alignment: find a good way to deal with this [remove the sequence from the inference if the insertion overlap with the delv range]
- test the restricted V gene option for generation.
- clean up gen event / static event if possible.
- add some checks so that people don't mix up the V and J files
- allow to fix number of cores used (```rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();``` -> make a function set_nb_threads(nb))
- extract_subsequence should pad with "BLANK" rather than with "N".
- righor-py, better "load_model" function
- problem in the CDR3 generation

## Other
- add cargo publish to the CI.yml
