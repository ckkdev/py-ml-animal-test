import json
from gensim.models import Word2Vec

# Load the animal data from an external JSON file
with open("animals.json", "r") as f:
    animal_data = json.load(f)['animals']

# Preprocess the animal names
processed_animals = [animal.lower().split() for animal in animal_data]

# Train the Word2Vec model
model = Word2Vec(processed_animals, vector_size=100, window=5, min_count=1, workers=4)

# Save the trained model
model.save("animal_vectors.model")

# Load the saved model
loaded_model = Word2Vec.load("animal_vectors.model")

# Find the most similar animals for each animal in the list
similar_animals_dict = {}
for animal in animal_data:
    animal_key = animal.lower().replace(" ", "_")
    similar_animals = loaded_model.wv.most_similar(animal.lower().split(), topn=5)
    similar_animals_dict[animal_key] = [sim_animal[0].replace("_", " ") for sim_animal in similar_animals]

# Save the results to a JSON file
with open("results.json", "w") as f:
    json.dump(similar_animals_dict, f, indent=2)
