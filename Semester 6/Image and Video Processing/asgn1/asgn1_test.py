from asgn1_lokesh_venkatesh import LinReg
import json

with open('asgn1_data_publish.json', 'r') as file:
        master_dataset = json.load(file)

all_weights = []

for i in range(len(master_dataset)):
    example_data = master_dataset[i]
    example_model = LinReg(example_data)
    example_model.fit(learning_rate=0.0001, n_iters=100000)
    all_weights.append(example_model.get_weights().item())
print(all_weights)

with open('asgn1/asgn1_test_lokesh.json', 'w') as outfile:
    json.dump(all_weights, outfile)