import sys
import os

# Add the parent directory to the path to be able to import Vectors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vectors.Vectors import VECTORES

def expand_vectors():
    """
    Expands the vectors in VECTORES to include nivel_27.
    nivel_27 is a 3x3x3 matrix of zeros.
    """
    new_vectores = {}
    for word, vector in VECTORES.items():
        if len(vector) == 2:
            nivel_3 = vector[0]
            nivel_9 = vector[1]
            nivel_27 = [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]
            new_vectores[word] = [nivel_3, nivel_9, nivel_27]
        else:
            new_vectores[word] = vector
    return new_vectores

def format_vector_entry(word, vector):
    """
    Formats a single vector entry for pretty printing.
    """
    nivel_3_str = str(vector[0])
    nivel_9_str = str(vector[1]).replace("], [", "],\n          [")
    nivel_27_str = str(vector[2]).replace("]], [[", "]],\n          [[")
    return f'    "{word}": [{nivel_3_str}, {nivel_9_str}, {nivel_27_str}]'

if __name__ == "__main__":
    expanded_vectors = expand_vectors()
    with open("c:/Users/p_m_a/Aurora/Trinity/Trinity-3/vectors/Vectors_expanded.py", "w", encoding="utf-8") as f:
        f.write("VECTORES = {\n")
        # Use an iterator to handle the last comma
        num_items = len(expanded_vectors)
        for i, (word, vector) in enumerate(expanded_vectors.items()):
            f.write(format_vector_entry(word, vector))
            if i < num_items - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("}\n")

    print("Expanded vectors written to c:/Users/p_m_a/Aurora/Trinity/Trinity-3/vectors/Vectors_expanded.py")
