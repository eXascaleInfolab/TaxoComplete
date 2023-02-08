This is the repositotry of TaxoComplete: Self-Supervised Taxonomy Completion Leveraging Position-Enhanced Semantic Matching

We use python 3.9 and cuda 11.1
Create a virtual environment using:

conda create -n taxocomplete python=3.9
python3.9 -m venv taxocomplete
We install pytorch via the following command:
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c dglteam dgl-cuda11.1
pip install -r requirements

This repository is organized as follows:
- The folder data contains the datasets used in the paper, where each contains data.taxo, data.terms and term2def:
    - MAG-CS-WIKI
    - MAG-PSY
    - SemEval-Noun
    - SemEval-Verb

