import pickle
import json

# Open the label_encoder.pickle file
with open("lstm_model/label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# Create a dictionary mapping each class index to its name
label_map = {i: label for i, label in enumerate(label_encoder.classes_)}

# Save the dictionary to a JSON file
with open("lstm_model/label_map.json", "w") as f:
    json.dump(label_map, f)

print("label_map.json has been created successfully.")
