This is the repositotry of TaxoComplete: Self-Supervised Taxonomy Completion Leveraging Position-Enhanced Semantic Matching

### Installation
- We use python 3.9 and cuda 11.1
- First, create a virtual environment using:
``` bash
conda create -n taxocomplete python=3.9
```
or 
``` bash
python3.9 -m venv taxocomplete
```
- Second, install pytorch and dgl via the following command:
``` bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c dglteam dgl-cuda11.1
```
- Finally, all remaining required packages could be installed with the requirements file:
``` bash
pip install -r requirements
```

### Repository Organization
This repository is organized as follows:
- data: contains the datasets used in the paper, where each contains dataset-name.taxo (each line is a pair of (parent node, child node) in the taxonomy), dataset-name.terms (each line is a pair of (node id, node name)) and term2def (each line is the node and its definition from a corresponding corpus):
    - MAG-CS-WIKI
    - MAG-PSY
    - SemEval-Noun
    - SemEval-Verb
- config_files: we specify for each dataset the hyperparameters used to train TaxoComplete.
- config_files_evaluate: we provide config files to evaluate on MAG-PSY-Wiki in case one has a trained model.
- src: contains the source code of TaxoComplete

### Running the code
- For the sake of reproducibility, we specify all hyperparameters in the config_files folder. For instance, to train and evaluate TaxoComplete on the MAG-PSY-Wiki dataset, use the following command:
``` bash
python ./src/train.py --config ./config_files/psy/config_clst20_s47.json
```
- If you have a trained model and would like to evaluate it, you can specify in a config_file the model path as shown config_files_evaluate and use the following command:
``` bash
python ./src/evaluate.py --config ./config_files_evaluate/psy/config_clst20_s49.json
```

## Citation
Please cite the following paper when using TaxoComplete:
``` bash
@inproceedings{arous2023www,
  title = {TaxoComplete: Self-Supervised Taxonomy Completion Leveraging Position-Enhanced Semantic Matching},
  author = {Arous, Ines, Ljiljana Dolamic and Cudr{\'e}-Mauroux, Philippe},
  booktitle = {Proceedings of the Web Conference (WWW 2023)},
  year = {2023},
  address = {Texas, USA}
}
```



