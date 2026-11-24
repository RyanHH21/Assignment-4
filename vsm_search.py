#-------------------------------------------------------------------------
# AUTHOR: Ryan Hoang
# FILENAME: vsm_search.py
# SPECIFICATION: This program builds a simple search engine using the
#                Vector Space Model. It turns the documents and queries into
#                TF-IDF vectors (removing only stopwords), then ranks all 100
#                documents for each query by cosine similarity. It uses the
#                relevance judgments to calculate the Average Precision of
#                each query, sorts the queries from best to worst AP, and
#                calculates the MAP to measure how well the search engine works.
# FOR: CS 5180- Assignment #4
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------
docs = pd.read_csv("docs.csv")                         
queries = pd.read_csv("queries.csv")                   
judgments = pd.read_csv("relevance_judgments.csv")     

relevant_docs = (judgments[judgments["judgment"] == "R"]
                 .groupby("query_id")["doc_id"]
                 .apply(set)
                 .to_dict())

# ---------------------------------------------------------
# 2. Build the TF-IDF matrix for the documents
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
doc_matrix = vectorizer.fit_transform(docs["text"])

# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------
ap_values = {}

for _, row in queries.iterrows():
    query_id = row["query_id"]
    query_vector = vectorizer.transform([row["query_text"]])
    scores = cosine_similarity(query_vector, doc_matrix)[0]
    ranking = (pd.DataFrame({"doc_id": docs["doc_id"], "score": scores})
               .sort_values(by="score", ascending=False, kind="mergesort")
               ["doc_id"].tolist())

    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------
    relevant = relevant_docs.get(query_id, set())
    relevant_found = 0
    precision_sum = 0

    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            relevant_found += 1
            precision_sum += relevant_found / rank    

    ap = precision_sum / len(relevant) if relevant else 0.0
    ap_values[query_id] = ap

# ---------------------------------------------------------
# 5. Sort queries by AP in descending order
# ---------------------------------------------------------
sorted_queries = sorted(ap_values.items(), key=lambda item: item[1], reverse=True)
map_score = sum(ap_values.values()) / len(ap_values)

# ---------------------------------------------------------
# 6. Print the sorted queries and their AP scores
# ---------------------------------------------------------
print("====================================================")
print("Queries sorted by Average Precision (AP):")
query_text = dict(zip(queries["query_id"], queries["query_text"]))
for position, (query_id, ap) in enumerate(sorted_queries, start=1):
    print(f"{position}. {query_id} ({query_text[query_id]}): AP = {ap:.4f}")

print(f"\nMean Average Precision (MAP): {map_score:.4f}\n")