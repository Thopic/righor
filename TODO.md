## Things to do:

- test the inference in detail
- add more tests
- Potential insertion in V/J alignment: find a good way to deal with this [remove the sequence from the inference if the insertion overlap with the delv range]
- test the restricted V gene option for generation.
- modify the way I deal with added error (make it cleaner, with a "ErrorDistribution" thing or smt)
- deal with amino-acid and generic "undefined" stuff.
  Strat: define an extended Dna object that the alignment can deal with +
  define the insertion thing so that it can deal with that
	  This second one is slightly a pain (the first one too ? No it's fine, just a bit longer to deal with).
		I would need to add sums here and there, nothing impossible, but slightly more a pain. In short some position must be linked, this will complexify quite a bit the definition of Dna (more precisely this will be a new class). So
	UndefinedDna would contains for each position a vec/array of bytes and a int giving the positions they're connected with  (just need two options for everything). This is very specific to the aa case, but why should I care. A bit complicated rn, leaving it for later.
- clean up gen event / static event if possible.
- add some checks so that people don't mix up the V and J files
- allow restricted models (P(V) P(DJ))
- allow to fix number of cores used (```rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();``` -> make a function set_nb_threads(nb))
- I should probably change the way I'm doing the inference (the feature object should contains ref to other features ?)


- righor-py, better "load_model" function
- problem in the CDR3 generation

Before version change checklist:
- is righor-py using the cargo package or the local version ?
- did you change the version number everywhere ?


## Structure

Ideally, each feature/parameter of the model is associated with 3/4 structures:
- a (list of) parameter(s) (for example p_vdj for the V,D,J distribution)
- a "feature", ie a list of parameters that can be easily updated / modified
- a generation struct that can generate fast what's needed
- a property/instance object that can contain a specific instance of the generation process
