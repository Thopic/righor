import righor
import pandas as pd
import os


os.environ["RUST_BACKTRACE"] = '1'

def read_fasta(filename):
    sequences = {}
    with open(filename, 'r') as file:
        current_header = None
        current_sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    sequences[current_header] = ''.join(current_sequence)
                current_header = line[1:]  # Remove ">" from the header
                current_sequence = []
            else:
                current_sequence.append(line)
        if current_header:
            sequences[current_header] = ''.join(current_sequence)
    return sequences

unproductive = read_fasta('test_righor.fasta')

print("Load model")
model = righor.load_model('human','igh')
align_params = righor.AlignmentParameters()
inference_params = righor.InferenceParameters()
als = model.align_all_sequences([unproductive[k] for k in unproductive], align_params)

print('define model with uniform parameters')
model = model.copy().uniform()

# infer model
print('infer model')
models = [model.copy()]
for rd in range(10):
    print(rd)
    model.infer(als, align_params=align_params, inference_params=inference_params)
    models.append(model.copy())