import os 

# Set the project directory.
project_dir = ''

# Set the directory to store triplets and the file path for the .csv containing them.
medmcqa_triplets_dir = os.path.join(project_dir, 'medmcqa_triplets')
medmcqa_triplets_csv_path = os.path.join(project_dir, 'medmcqa_triplets.csv')

# Set the output directory for model generations.
medmcqa_output_dir = os.path.join(project_dir, 'medmcqa_outputs')

# Set the output path for the entities and embeddings. 
medmcqa_entities_path = os.path.join(project_dir, 'medmcqa_entities.pkl')
medmcqa_embeddings_path = os.path.join(project_dir, 'medmcqa_embeddings.npy')

# To reproduce the results of the paper only a HuggingFace access token is required.
openai_api_key = ''
hf_access_token = ''

# The list of evaluated models in this study, some of them require access (see their official HuggingFace pages).
all_models = [
    'meta-llama/Llama-3.1-8B-Instruct',
    'google/gemma-2-9b-it',
    'CohereForAI/c4ai-command-r7b-12-2024',
    'mistralai/Ministral-8B-Instruct-2410',
    'Falcon3-10B-Instruct'
]

# Set the Neo4j database credentials.
uri, db_name, user, password = 'neo4j://localhost:7687', 'neo4j', 'neo4j', '12345678'
