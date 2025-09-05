from asgn2_Gurjot import LinReg
import json

with open('asgn1_data_publish.json', 'r') as file:
        master_dataset = json.load(file)

all_weights = []

for i in range(len(master_dataset)):
    example_data = master_dataset[i]
    example_model = LinReg(example_data)
    example_model.fit()
    all_weights.append(example_model.get_weights().item())

with open('asgn2/asgn2_test_gurjot.json', 'w') as outfile:
    json.dump(all_weights, outfile)