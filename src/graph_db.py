import sys
import numpy
import pathlib

from typing import Any, TypeVar, Self
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, DriverError, ServiceUnavailable
from src.vector_index import search, rank_triplets_with_embeddings

EmbeddingModel = TypeVar('EmbeddingModel')
Neo4jDriver = TypeVar('Neo4jDriver')


class Neo4jDatabase(object): 
    """
    A wrapper class which manages the lifecycle of the graph database driver object.
    """
    
    def __init__(self: Self, uri: str, db_name: str, user: str, password: str): 
        """
        Initialization method.

        Arguments
        ---------
        uri: the database unique resource identifier (str).  
        db_name: the database name (str).  
        user: the database user (str).  
        password: the database password (str).  
        """

        # Create the database driver and initialize the connection.
        self.driver = GraphDatabase.driver(uri, auth = (user, password), database = db_name)
        self.db_name = db_name

        # Warmup the connection by sending the first query (requires a few seconds).
        with self.driver.session() as session:
            try:
                session.run('CALL db.info')
            except (Neo4jError, DriverError, ServiceUnavailable) as e:
                print(e, file = sys.stderr)
                sys.exit(0)


    def run(self: Self, query: str, **kwargs: Any) -> list[list[Any]]:
        """
        A method which runs queries in the Neo4j database by opening and closing a session.
        Managing the session has much lower computational complexity than running a query.

        Arguments
        ---------
        query: a parameterized cypher query (str).  
        kwargs: Any query parameters to be passed as arguments in the session.run() method (Any).  
        
        Returns
        -------
        A list of records with their values being lists.
        """
        records = None
        with self.driver.session(database = self.db_name) as session:
            try:
                result = session.run(query, **kwargs)
                records = [tuple(record.values()) for record in result]
            except (Neo4jError, DriverError) as e:
                print(e, file = sys.stderr)

        return records


    def close(self: Self):
        """
        A method which closes the database connection.
        """
        self.driver.close()


def store_triplets_in_graph(driver: Neo4jDriver, csv_path: str):
    """
    Function which transforms triplets from a .csv into a graph form 
    and stores them in Neo4j using cypher statements.

    Arguments
    ---------
    driver: the database driver object (Neo4jDriver).
    csv_path: file path to save the .csv (str).

    Returns
    -------
    None.
    """
    # Create uniqueness constraints to avoid duplicate information in the knowledge graph.
    query = ('CREATE CONSTRAINT entity_name FOR (e:Entity) REQUIRE e.name IS UNIQUE')
    driver.run(query)

    # Convert the file path to a file uri.
    file_uri = pathlib.Path(csv_path).as_uri()

    # Load subjects and objects from the .csv and store them as unique nodes. 
    query = (
        f'LOAD CSV WITH HEADERS FROM "{file_uri}" AS row '
        'WITH row '
        'WHERE row.subject IS NOT NULL '
        'AND row.object IS NOT NULL '
        'AND row.predicate IS NOT NULL '
        'MERGE (s:Entity {name: row.subject}) '
        'MERGE (o:Entity {name: row.object}) '
        'MERGE (s)-[p:HAS_PREDICATE {name: row.predicate}]->(o)'
    )
    driver.run(query)

    return


def graph_triplet_search(
        driver: Neo4jDriver, embedding_model: EmbeddingModel, 
        search_terms: list[str], entities: list[str], 
        embeddings: numpy.ndarray, question: str) -> str:
    """
    Function which utilizes a list of extracted terms,
    to search the knowledge graph (Neo4j) for relevant triplets using various methods.
    The triplets are returned as newline separated string.

    Arguments
    ---------
    driver: the database driver object (Neo4jDriver).
    embedding_model: the embedding model (EmbeddingModel).
    search_terms: the list of search terms extracted from the LLM (list[str]).
    entities: the list of all entities from the csv (list[str]).
    embeddings: the numpy array containing all embeddings (numpy.ndarray)
    question: the question to be answered (str).

    Returns
    -------
    contextual_triplets: the newline separated string that comprises a list of relevant triplets; 
                         used as context by the LLM (str).
    """
    
    # Lowercase the search terms.
    lowercased_terms = list(map(str.lower, search_terms))

    # Declare some important queries to be used below.
    exact_match_query = (
        'UNWIND $search_terms AS entity_name '
        'MATCH (e: Entity {name: entity_name})-[r:HAS_PREDICATE]->(o: Entity) '
        'RETURN e.name, r.name, o.name'
    )

    # Find triplets using exact term match.
    triplets = set(driver.run(exact_match_query, search_terms = lowercased_terms))

    # Search the similarity embeddings index to find highly similar entities.
    highly_similar_entities = search(
        embedding_model, entities, embeddings, lowercased_terms
    )

    # Find triplets based on the above entities.
    similarity_triplets = driver.run(
        exact_match_query, search_terms = highly_similar_entities
    )
    triplets.update(similarity_triplets)

    # Rank the triplets based on descending similarity score using embeddings.
    triplets = rank_triplets_with_embeddings(embedding_model, triplets, question)

    # Join the relevant triplets into a context string.
    contextual_triplets = '\n'.join(' '.join([element for element in triplet]) for triplet in triplets)

    return contextual_triplets
