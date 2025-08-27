from src.config import *
from src.experiments import (
    first_time_setup,
    medmcqa_model_generations,
    construct_all_generated_csvs,
    evaluate_all_generated_csvs_on_medmcqa,
)

def main():

    # Run the first time setup.
    first_time_setup()

    # Run model generations for the datasets.
    medmcqa_model_generations(medmcqa_output_dir, medmcqa_entities_path, medmcqa_embeddings_path)

    # Construct all generated csvs from invidual text files.
    construct_all_generated_csvs(medmcqa_output_dir)

    # Evaluate all generated csvs using the ground truth labels.
    evaluate_all_generated_csvs_on_medmcqa(medmcqa_output_dir)

    return


if __name__ == '__main__': main()
