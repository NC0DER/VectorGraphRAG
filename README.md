[![Python-Versions](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![Software-License](https://img.shields.io/badge/License-Apache--2.0-green)](https://github.com/NC0DER/VectorGraphRAG/blob/main/LICENSE)

# VectorGraphRAG

This repository hosts code for the paper:
* [Giarelis, N., Mastrokostas, C., & Karacapilidis, N. (2025). VectorGraphRAG: Automatic Knowledge Graph Construction and Memory Efficient Triplet Ranking for Medical Question Answering]()


## Installation
This projects requires a properly configured Neo4j database (version >= 5.24) and the following python packets installed.
```
pip install requirements.txt
```

## Reproducibility
To reproduce the results of this paper, use the `GPT4o-mini` extracted triplets stored in `medmcqa_triplets.csv`.
Some of the evaluated open-weights models require an access request from their official HuggingFace pages (e.g., Llama-3.1-8B-instruct, Gemma-2-9B-it).
Once you are granted access to these models, set your huggingface access token in the respective variable of `config.py` before running the experiments.

## Citation
The model has been officially released with the article:  
[VectorGraphRAG: Automatic Knowledge Graph Construction and Memory Efficient Triplet Ranking for Medical Question Answering]().  
If you use the data, code or model, please cite the following:

```bibtex
TBA
```

## Contributors
* Nikolaos Giarelis (giarelis@ceid.upatras.gr)
* Charalampos Mastrokostas (cmastrokostas@ac.upatras.gr)
* Nikos Karacapilidis (karacap@upatras.gr)
