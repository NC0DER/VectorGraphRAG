import os
import pandas

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from tqdm import tqdm

from src.config import *
from src.llms import LLM, LLM_prompt
from src.utils import (
    extract_search_terms,
    extract_letter_from_generated_answer
)
from src.triplets import construct_triplets_csvs
from src.vector_index import (
    create_entities_embeddings_files, 
    load_entities_and_index
)
from src.graph_db import (
    Neo4jDatabase, 
    store_triplets_in_graph,
    graph_triplet_search
)

def first_time_setup():

    # Construct the triplets and save them to .csv files.
    # construct_triplets_csvs()

    # Load the sentence transformers embeddings model.
    embedding_model = SentenceTransformer('all-mpnet-base-v2', device = 'cpu')

    # Create the vector index from the triplets and save them into binary file.
    # The sorted list of all entities are also saved into a binary file.
    create_entities_embeddings_files(
        embedding_model, medmcqa_triplets_csv_path, 
        medmcqa_entities_path, medmcqa_embeddings_path
    )

    # Create the database driver object and initialize the connection.
    driver = Neo4jDatabase(uri, db_name, user, password)

    # Load the triplets and store them into a knowledge graph.
    # For different datasets you need a different database instance running in Neo4j Desktop.
    store_triplets_in_graph(driver, csv_path = medmcqa_triplets_csv_path)

    return


def construct_generated_csv(dir_name: str, dir_path: str):
    """
    Function that combines multiple generated outputs from .txt files
    and stores them in a single .csv.

    Parameters
    -----------
    dir_name: the name of the directory (str).
    dir_path: the path of the directory where generated outputs are stored (str).

    Returns
    --------
    None
    """

    # Construct the input path.
    path = Path(dir_path)

    # Find all .txt files and their assosiated paths.
    txt_paths = [
        p.absolute()
        for p in path.iterdir()
        if p.is_file() and p.suffix == '.txt'
    ]

    # Append all generated outputs to a list.
    generated_outputs_list = []

    for txt_path in tqdm(txt_paths, desc = 'Joining text generations into a single .csv'):
        # Each text files contains exactly one generated output.
        generated_output = ''
        with open(txt_path, 'r', encoding = 'utf-8', errors = 'ignore') as f:
            generated_output = f.read()

        # Get the filename (dataset id) from the path.
        id = Path(txt_path).stem

        generated_outputs_list.append([id, generated_output])

    # Make a dataframe which stores all generated outputs.
    df = pandas.DataFrame(
       generated_outputs_list,
       columns = ['id', 'generated_outputs']
    )

    # Save the dataframe into a .csv dataset in the same directory.
    df.to_csv(os.path.join(dir_path, f'{dir_name}.csv'), encoding = 'utf-8', index = False)

    return


def construct_all_generated_csvs(generated_outputs_dir: str):
    """
    Wrapper function which accesses all directories that have generated 
    output files and produces a .csv with all combined answers for each one.
    This .csv is stored in the same directory as the combined answers.

    Parameters
    -----------
    generated_outputs_dir: the directory containing multiple subdirectories each containing text generations.

    Returns
    --------
    None
    """
    # Find all subdirectories.
    subdirectories = [(f.name, f.path) for f in os.scandir(generated_outputs_dir) if f.is_dir()]

    # Construct the generated .csv for each subdirectory.
    for subdir in tqdm(subdirectories):
        construct_generated_csv(*subdir)

    return


def evaluate_model_on_medmcqa(dir_name: str, dir_path: str):
    """
    Function which evaluates a single LLM, by comparing its selected answers
    against the ground truth answers of the MedMCQA dataset.
    These selected answers are extracted from the text generations of the LLM.
    The evaluation metric used is accuracy (%).

    Parameters
    -----------
    dir_name: the name of the directory (str).
    dir_path: the path of the directory where generated outputs are stored (str).

    Returns
    --------
    None
    """

    # Get all text generations of the LLM and sort them based on id.
    df = pandas.read_csv(os.path.join(dir_path, f'{dir_name}.csv'), index_col = False)

    # Load the validation split of the MedMCQA dataset with single option questions.
    medmcqa_val = load_dataset('openlifescienceai/medmcqa', split = 'validation')
    medmcqa_val = medmcqa_val.filter(lambda x: x['choice_type'] == 'single')
    
    # Sort the generated answers and dataset by id to align them.
    df = df.sort_values('id')
    medmcqa_val = medmcqa_val.sort('id')

    model_answers, ground_truth_labels = [], []
    for gen_id, output, row in zip(df['id'], df['generated_outputs'], medmcqa_val):

        # Map the numeric label to the corresponding letter.
        ground_truth_label = str(row['cop']).translate(str.maketrans('0123', 'ABCD'))

        if gen_id != row['id']: raise Exception('This should not happen!')

        # Retrieve the options for this row.
        options = [row['opa'], row['opb'], row['opc'], row['opd']]

        # Join the possible options with double newline characters.
        options_text = '\n\n'.join([
            f'{letter}. {option}' 
            for letter, option in zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', options)
        ])

        # Use the regexpr pattern to search for the correct letter.
        model_answer = extract_letter_from_generated_answer(output, options_text, 'A-D')

        ground_truth_labels.append(ground_truth_label)
        model_answers.append(model_answer)
    
    # Calculate the accuracy score based on these two lists.
    results = accuracy_score(ground_truth_labels, model_answers)
    print(f'{dir_name}: {round(results * 100, 2)}')
    
    return


def evaluate_all_generated_csvs_on_medmcqa(generated_outputs_dir: str):
    """
    Wrapper function which accesses all directories that have the generated 
    output .csv for each model and runs the evaluation function.

    Parameters
    -----------
    generated_outputs_dir: the directory containing multiple subdirectories each containing text generations.

    Returns
    --------
    None
    """
    # Find all subdirectories.
    subdirectories = [(f.name, f.path) for f in os.scandir(generated_outputs_dir) if f.is_dir()]

    # Evaluate the outputs of each model.
    for subdir in tqdm(subdirectories):
        evaluate_model_on_medmcqa(*subdir)

    return


def medmcqa_model_generations(output_dir: str, entities_path: str, embeddings_path: str):

    # Set (and create) the directory (if it does not already exist) for the generated model outputs.
    Path(output_dir).mkdir(parents = True, exist_ok = True)

    # Load the validation split of the MedMCQA dataset with the single option questions.
    medmcqa_val = load_dataset('openlifescienceai/medmcqa', split = 'validation')
    medmcqa_val = medmcqa_val.filter(lambda x: x['choice_type'] == 'single')

    # Initialize database credentials.
    uri, db_name, user, password = 'neo4j://localhost:7687', 'neo4j', 'neo4j', '12345678'

    # Create the database driver object and initialize the connection.
    driver = Neo4jDatabase(uri, db_name, user, password)

    # Load the sentence transformers embeddings model.
    embedding_model = SentenceTransformer('all-mpnet-base-v2', device = 'cpu')

    # Load the entities and embeddings from the disk.
    entities, embeddings = load_entities_and_index(entities_path, embeddings_path)

    # Declare the system prompt for all models.
    system_prompt = (
        'You are an informative chatbot. '
        'Please answer the user\'s question to the best of your ability. '
        'If you do not know something please state that to the user.'
    )

    for model_name in all_models:

        # Load a new model in each iteration.
        model = LLM(model_name, 42)

        # Extract the model name without the organization repository.
        model_file_name = model_name.split('/')[-1]

        for i, row in tqdm(enumerate(medmcqa_val)):

            # Extract all potential options from the row.
            options = [row[column] for column in ['opa', 'opb', 'opc', 'opd']]

            # Generate the answer using only the internal knowledge of the LLM.
            # This is the baseline answer of this experiment for the selected model.
            generated_answer = LLM_prompt(
                model, system_prompt, '', 
                row['question'], options, 
                use_context = False
            )

            # Create the directory (if it does not exist) for the baseline generated answers.
            save_generated_dir = os.path.join(output_dir, f'{model_file_name}_Baseline')
            Path(save_generated_dir).mkdir(parents = True, exist_ok = True)

            # Save each generated response to a separate file.
            with open(
                os.path.join(
                    save_generated_dir, f"{row['id']}.txt"
                ), 'w', encoding = 'utf-8', errors = 'ignore') as f:
                f.write(generated_answer)

            # Extract search terms from the question and its possible options.
            search_terms = extract_search_terms(' '.join([row['question'], *options]))

            # Search the graph for relevant triplets containing the extracted search terms. 
            contextual_triplets = graph_triplet_search(
                driver, embedding_model, search_terms, 
                entities, embeddings, row['question']
            )

            # Generate the answer using the LLM and the context provided by the KG.
            generated_answer = LLM_prompt(
                model, system_prompt, contextual_triplets, 
                row['question'], options, use_context = True
            )

            # Create the directory (if it does not exist) for the generated answers based on the current method.
            save_generated_dir = os.path.join(output_dir, f'{model_file_name}_VectorGraphRAG')
            Path(save_generated_dir).mkdir(parents = True, exist_ok = True)

            # Save each generated response to a separate file.
            with open(
                os.path.join(
                    save_generated_dir, f"{row['id']}.txt"
                ), 'w', encoding = 'utf-8', errors = 'ignore') as f:
                f.write(generated_answer)

    return
