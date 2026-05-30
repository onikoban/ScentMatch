from Embedding_Model.retrieval import Retrieval
from Embedding_Model.generation import generate_response


query = input("Enter query: ")

results = Retrieval(query)

response = generate_response(query, results)

print(response)