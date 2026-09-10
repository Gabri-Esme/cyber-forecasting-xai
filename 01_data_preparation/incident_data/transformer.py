import os
import re
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture

# Utilities

def load_csv(file_path):
    '''Loads a CSV file into a pandas DataFrame.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    '''
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    return df

def combine_text_from_columns(df, columns):
    '''
    Combines text from specified columns in a DataFrame into a single string for each row.
    Args:
        df (pd.DataFrame): The input DataFrame containing the text data.
        columns (list): A list of column names to combine.
    Returns:
        pd.Series: A Series containing the combined text for each row.
    '''
    combined_text = df[columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    return combined_text 


def redact_and_clean_texts(texts):
    ''' Function to redact geographic and political entities in a collection of texts. Additionally, prepares formatting for natural language processing
    Args: 
        texts (list): List of text strings for redaction
    Returns: 
        cleaned_texts (list) : List of sanitised texts'''

    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        nlp = spacy.load("en_core_web_sm")
    
    cleaned_redacted_texts = []
    disabled_components = ["tagger", "parser", "attribute_ruler", "lemmatizer"]
    
    for doc in nlp.pipe(texts, batch_size=250, disable=disabled_components):
        tokens = []
        for token in doc:
            if token.ent_type_ in {"GPE", "LOC", 'NORP', 'FAC'}:
                if not tokens or tokens[-1] != '[REDACTED]':
                    tokens.append('[REDACTED]')
            else:
                tokens.append(token.text)
                
        text = " ".join(tokens).lower()
        text = re.sub(r'\s+', ' ', text).strip()
        cleaned_redacted_texts.append(text)
   
    return cleaned_redacted_texts

# Embedding
def embed_text(texts, model_name='all-MiniLM-L6-v2'):
    '''
    Embeds the input texts.
    Args:
        texts (list): A list of strings to be embedded.
        model_name (str): The name of the SentenceTransformer model to use for embedding.
    Returns:
        df: Dataframe containing embedding reference (to later match backl to original dataset) and corresponding embedding)
    '''
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    np.save('01_data_preparation/incident_data/gaussian_clusters/embeddings.npy', embeddings)

    return embeddings

def embedding_umap_reduction(df, n_dims=int):
        '''Function to create a 2d visual of clustering prevalence
        Args:
            df: data frame containing embeddings'''
        reducer_dim_umap = umap.UMAP(
        n_neighbors=15,
        min_dist=0.0, 
        n_components=n_dims, 
        random_state=42
        )
        embeddings_reduced = reducer_dim_umap.fit_transform(df)
        np.save(f'01_data_preparation/incident_data/gaussian_clusters/embeddings_{n_dims}d.npy', embeddings_reduced)
        return embeddings_reduced

# Clustering
def gaussian_clusterer(orig_df, embedding_df, n_clusters=15):
    '''Takes the original dataset and embeddings, finds clusters based on the embeddings df.
    Then, a column is added for each cluster and filled with a score, which represnets the gaussian probability of belonging to a cluster
    Finally, all the information is saved in a new csv
    Args: 
        orig_df: df 
        embedding_df: df
        n_clusters: number of clusters
    Returns
        csv'''
    clusterer = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=42, n_init=1, max_iter=50)
    clusterer.fit(embedding_df)
    probabilities = clusterer.predict_proba(embedding_df)
    scores = np.floor(probabilities*10).astype(int)


    score_cols = [f'cluster_{i}_score' for i in range(n_clusters)]
    df_scores = pd.DataFrame(data=scores, columns=score_cols)

    results_df = pd.concat([orig_df.reset_index(drop=True), df_scores], axis=1)
    results_df.to_csv('01_data_preparation/incident_data/output_data/incidents_clustered.csv') 

    return probabilities

def compute_centroids(probabilities, raw_embeddings, n_clusters):
    centroids = []
    for i in range(n_clusters):
        weights = probabilities[:, i]
        sum_w = weights.sum()
        if sum_w >0:
            centroid = np.dot(weights, raw_embeddings) / sum_w
        else:
            centroid = np.zeros(raw_embeddings.shape[1])
        centroids.append(centroid)

    centroid_matrix = np.array(centroids)
    np.save('01_data_preparation/incident_data/gaussian_clusters/cluster_centroids.npy', centroid_matrix)


def orchestrator(file_path, columns):
    '''Orchestrator function, takes file_path of orginal CSV an columns of interest.
    Loads, cleans, applies embedding model, reduces dimensionality and creates clusters. 
    Final output is original csv with addition of embeddigns and clusters with score
    Args:
        file_path: csv
        columns: columns to be merged'''
    # Load and prepare data
    df = load_csv(file_path)

    combined_texts = combine_text_from_columns(df, columns)
    redacted_texts = redact_and_clean_texts(combined_texts)

    # Embed text
    embeddings = embed_text(redacted_texts)
    df['embedding_reference'] = ['embedding_' + str(i) for i in range(len(df))]

    embeddings_2d = embedding_umap_reduction(embeddings, n_dims=2)
    embeddings_10d = embedding_umap_reduction(embeddings, n_dims=10)

    # Cluster and save df
    n_clusters = 15
    probabilities = gaussian_clusterer(df, embeddings_10d, n_clusters)

    # Compute centroids for evaluation 
    compute_centroids(probabilities, embeddings, n_clusters)

if __name__ == "__main__":
    orchestrator("01_data_preparation/incident_data/raw_data/eurepoc_data_2026-09-05T12_09.xlsx", (["name", "description"]))

