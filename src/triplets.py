import os
import pandas
import pathlib
from src.config import *
from src.gpt4oai import GPT4oAI
from datasets import load_dataset
from tqdm import tqdm


def extract_medmcqa_triplets(dir_path: str):
    """
    Function that extract triplets from each training sample
    using the GPT4o-mini model and saves them in a file.

    Parameters
    -----------
    dir_path: the path of the directory to store triplets in (str).

    Returns
    --------
    None.
    """

    # Make the folder if it does not already exists.
    pathlib.Path(dir_path).mkdir(exist_ok = True)

    # Initialize the GPT4o client.
    gpt4o_model = GPT4oAI()

    # Load the medmcqa dataset, select the appropriate subset and sort it.
    medmcqa = load_dataset('openlifescienceai/medmcqa', split = 'train')
    single_medmcqa = medmcqa.filter(lambda x: x['choice_type'] == 'single')
    print(f'MCQA: {len(single_medmcqa)}')

    for i, row in enumerate(tqdm(single_medmcqa)):

        # Get the correct answer index for each question.
        correct_index = str(row['cop'])
        correct_index = correct_index.translate(str.maketrans('0123', 'abcd'))
        correct_index = ''.join(('op', correct_index))

        # Join the question and the correct answer to get the complete training sample.
        training_sample = ' '.join(('Question: ', row['question'], '\nAnswer: ', row[correct_index]))

        # Construct the prompt using the training sample.
        prompt = (
            'Please extract a semantic triplet in the form (subject, predicate, object) '
            'from the following question answer pair. '
            'Use the Answer in your triplet.\n'
            f'{training_sample}\n'
            'Output only the triplet itself.'
        )
        
        # Infer the triplet from the model and save it in a file.
        triplet = gpt4o_model.infer(prompt)
        with open(
            os.path.join(dir_path, f'{row["id"]}.txt'), 
            'w', encoding = 'utf-8', errors = 'ignore') as f:
            f.write(triplet)

    return


def combine_multiple_triplet_csvs(input_path: str, output_path: str):
    """
    Function that combines multiple triplets from .txt files
    and stores them in a single .csv.

    Parameters
    -----------
    input_path: the path of the directory where triplets are stored (str).
    output_path: the path of the .csv file (str).

    Returns
    --------
    None
    """

    # Construct the input path.
    path = pathlib.Path(input_path)

    # Find all .txt files and their assosiated paths.
    txt_paths = [
        p.absolute()
        for p in path.iterdir()
        if p.is_file() and p.suffix == '.txt'
    ]

    # Append all triplets to a list.
    triplets_list = []
    empty_triplets = 0
    malformed_triplets = 0

    for i, txt_path in enumerate(tqdm(txt_paths)):
        # Each text files contains exactly one triplet.
        triplet = ''
        with open(txt_path, 'r', encoding = 'utf-8', errors = 'ignore') as f:
            triplet = f.read()
        
        # If the triplet does not exist, continue to the next row.
        if not triplet:
            empty_triplets += 1
            continue

        # If the triples were malformed, because of the LLM, continue to the next row.
        if triplet.count(', ') != 2:
           malformed_triplets += 1
           continue

        # Remove extra whitespace and parentheses.
        triplet = ' '.join(triplet.split())
        triplet = triplet.replace('(', '')
        triplet = triplet.replace(')', '')

        # Separate the triplet into its parts and append it to the list without leading whitespaces.
        subj, pred, obj = triplet.split(', ')
        triplets_list.append([subj.strip(), pred.strip(), obj.strip()])

    print(f'Empty triplets: {empty_triplets}')
    print(f'Malformed triplets: {malformed_triplets}')

    # Make a dataframe which stores all triplets.
    df = pandas.DataFrame(
       triplets_list,
       columns = ['subject', 'predicate', 'object']
    )

    # Save the dataframe into a .csv dataset.
    df.to_csv(output_path, encoding = 'utf-8', index = False)

    return


def construct_triplets_csvs():

    # Extract triplets from MedMCQA using GPT4o and combine them into a single .csv.
    extract_medmcqa_triplets(medmcqa_triplets_dir)
    combine_multiple_triplet_csvs(medmcqa_triplets_dir, medmcqa_triplets_csv_path)

    return


if __name__ == '__main__': construct_triplets_csvs()
