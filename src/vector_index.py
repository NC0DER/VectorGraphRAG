import numpy
import faiss
import pandas
import pickle

from typing import TypeVar
from sentence_transformers import SentenceTransformer

EmbeddingModel = TypeVar('EmbeddingModel')

def create_entities_embeddings_files(
        embedding_model: EmbeddingModel, csv_path: str, 
        entities_path: str, embeddings_path: str
    ):
    """
    A function which creates and stores two binary files. 
    These contain the set of unique KG entities and their embeddings.
    This is done by reading the triplets.csv and extract the entire set of entities.
    This set is encoded using a sentence transformers embedding model.

    Arguments
    ---------
    embeddings_model: The sentence transformers embeddings model (EmbeddingsModel).
    csv_path: The input .csv filepath (str).
    entities_path: The output path for the binary file containing the list of entities (str).
    embeddings_path: The output path for the numpy array file that comprises the embeddings index (str).
    
    Returns
    -------
    None.
    """
    # Read the .csv as a pandas dataframe.
    df = pandas.read_csv(csv_path, index_col = False)

    # Retrieve the set of entities and sort them into a list%.
    triplet_subjects = set(map(str, df['subject'].to_list()))
    triplet_objects = set(map(str, df['object'].to_list()))
    entities = sorted(triplet_subjects | triplet_objects)

    # Encode each entity using sentence-transformers and normalize them using L2.
    embeddings = embedding_model.encode(entities)
    faiss.normalize_L2(embeddings)

    # Save the entities to a pickle file and the embeddings as a numpy array file.
    with open(entities_path, 'wb') as f:
        pickle.dump(entities, f)
    numpy.save(embeddings_path, embeddings)

    return


def load_entities_and_index(
        entities_path: str, embeddings_path: str
    ) -> tuple[list[str], numpy.ndarray]:

    """
    A function which loads binary files containing 
    entities and their embeddings. 

    Arguments
    ---------
    entities_path: The output path for the binary file containing 
                   the list of entities (str).
    embeddings_path: The output path for the numpy array file 
                     that comprises the embeddings index (str).
    
    Returns
    -------
    <object>: The entities and their embeddings (tuple[list[str], numpy.ndarray]).
    """
    entities = []
    with open(entities_path, 'rb') as f:
        entities = pickle.load(f)
    embeddings = numpy.load(embeddings_path)
    return (entities, embeddings) 


def search(
        embeddings_model: EmbeddingModel, entities: list[str], embeddings: numpy.ndarray, 
        terms: list[str], top_n: int = 10, similarity_cutoff: float = 0.90
    ) -> list[str]:
    """
    A function which rebuilts the vector index from the entities and their embeddings. 
    The terms to be searched are similarly encoded to be used as queries for the index.

    Arguments
    ---------
    embeddings_model: The sentence transformers embeddings model (EmbeddingsModel).
    index_dict: The index dictionary (dict[str, list]).
    terms: the terms to be searched (list[str]).
    top_n: the top_n most similar terms (int).
    similarity_cutoff: the cutoff similary value (float). 
    
    Returns
    -------
    highly_similar_terms: a list of highly similar terms (list[str]).
    """

    # Encode the terms to get their embeddings and normalize them.
    term_embeddings = embeddings_model.encode(terms)
    faiss.normalize_L2(term_embeddings)

    # Find the embedding dimension.
    embedding_dim = len(embeddings[0])

    # Create the inner product index using the saved embeddings.
    # This is equivalent to a cosine similarity index given normalized embeddings.
    index = faiss.index_factory(embedding_dim, 'IDMap,Flat', faiss.METRIC_INNER_PRODUCT)

    # Create the ids from the original embeddings list.
    unranked_ids = numpy.array(range(len(embeddings))).astype(numpy.int64)

    # Add the embeddings and their ids to the index.
    index.add_with_ids(embeddings, unranked_ids)

    # Search the index using the term embeddings.
    # We search multiple embeddings at once due to a matrix optimization implemented in faiss.
    similarities_index, ranked_ids_index = index.search(term_embeddings, top_n)

    # We iterate the similarities and ranked ids indices for each term.
    # This assumes that the order of terms (and their embeddings) is preserved in the indices.
    highly_similar_terms = {
        similar_term
        for similarities, ranked_ids, term in zip(
            similarities_index, ranked_ids_index, terms
        )
        for similarity, ranked_id in zip(similarities, ranked_ids)
        if similarity >= similarity_cutoff 
        and (similar_term := entities[ranked_id]) != term
    }
    return list(highly_similar_terms)


def rank_triplets_with_embeddings(
        embeddings_model: EmbeddingModel, triplets_set: set[str], question: str, 
        top_n: int = 300, similarity_cutoff: float = 0.10
    ) -> list[str]:
    """
    A function which builds a vector index from the embeddings of the triplets, 
    in order to rank them in descending cosine similarity order with the 
    embedding of the question and return the top_n. 
    The triplets and questions are encoded with the same embedding model.

    Arguments
    ---------
    embeddings_model: The sentence transformers embeddings model (EmbeddingsModel).
    triplets_set: the set of triplets to be searched (set[str]).
    top_n: the top_n most similar triplets (int).
    similarity_cutoff: the cutoff similary value (float). 
    
    Returns
    -------
    highly_similar_triplets: a list of highly similar triplets to the question (list[str]).
    """
    # If the triplet set is empty, then early exit.
    if not triplets_set:
        return []

    # Convert the set to a list of triplets.
    triplets = list(triplets_set)

    # Encode the triplets and question to get their embeddings and normalize them.
    triplet_embeddings = embeddings_model.encode(triplets)
    question_embedding = numpy.atleast_2d(
        embeddings_model.encode(' '.join(question.split()))
    )
    faiss.normalize_L2(triplet_embeddings)
    faiss.normalize_L2(question_embedding)

    # Find the embedding dimension.
    embedding_dim = len(triplet_embeddings[0])

    # Create the inner product index using the newly created embeddings.
    # This is equivalent to a cosine similarity index given normalized embeddings.
    index = faiss.index_factory(embedding_dim, 'IDMap,Flat', faiss.METRIC_INNER_PRODUCT)

    # Create the ids from the embeddings list.
    unranked_ids = numpy.array(range(len(triplet_embeddings))).astype(numpy.int64)

    # Add the embeddings and their ids to the index.
    index.add_with_ids(triplet_embeddings, unranked_ids)

    # Search the index using the question embedding.
    similarities_index, ranked_ids_index = index.search(question_embedding, top_n)

    # We iterate the similarities and ranked ids indices for each triplet.
    # This assumes that the order of triplets (and their embeddings) is preserved in the indices.
    highly_similar_triplets = {
        triplets[ranked_id]
        for similarities, ranked_ids in zip(
            similarities_index, ranked_ids_index
        )
        for similarity, ranked_id in zip(similarities, ranked_ids)
        if similarity >= similarity_cutoff
    }
    return list(highly_similar_triplets)
