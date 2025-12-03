import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Define our "Concepts" as Vectors (the Knowledge Base)
# Format: [Can_it_be_pet, Is_it_furry]
knowledge_base = {
    "Dog": np.array([0.9, 0.9]),  # High pet, High fur
    "Cat": np.array([0.8, 0.8]),  # High pet, High fur
    "Wolf": np.array([-0.5, 0.8]),  # Low pet, High fur
    "Salad": np.array([-0.9, -0.9]),  # Not pet, Not fur
}

# 2. Define a "User Query"
# The user searches for: "I want a healthy green lunch"
# Translation to numbers: High pet (1.0), High fur (1.0)
query_vector = np.array([[-1.0, -1.0]])

# 3. the Search Engine (RAG Logic)
print(f"Use Query Vector: {query_vector}")
print("-" * 30)

for name, vector in knowledge_base.items():
    # Reshape vector to be 2D (1 row, 2 columns) for the math function
    db_vector = vector.reshape(1, -1)

    # Calculate Cosine Similarity
    # This measures the angle between the two arrows.
    # 1.0 = Identical direction (Perfect Match)
    # 0.0 = 90 degree apart (Unrelated)
    # -1.0 = Opposite direction (Opposite meaning)
    similarity = cosine_similarity(query_vector, db_vector)[0][0]

    print(f"Similarity to {name}: {similarity:.4f}")
