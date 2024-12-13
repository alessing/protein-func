## Setup

The GO Annotation dataset can be downloaded from this link: https://current.geneontology.org/products/pages/downloads.html. Download the Homo Sapien data.

The AlphaFold v2 structure predictions can be downloaded from this link: https://alphafold.ebi.ac.uk/download#proteomes-section. Download the Homo Sapien data.

To create the environment in which to run the model, run the following:

`conda env create -f environment.yml`

Then activate the environment:

`conda activate protein-func`

Then you can do a training run with default hyperparameters by running python `main.py`.

Here are the runs we did:

#1 `python main.py --lora_dim 16 --hidden_dim 256`

#2 `python main.py --model_type gat --hidden_dim 256`

#3 `python main.py --lora_dim 16 --hidden_dim 256 --use_conf_score False`

#4 `python main.py --lora_dim 16 --hidden_dim 256 --data_dir data/processed_data/hdf5_files_d_10 --feature_dim 14`

#5 `python main.py --lora_dim 16 --hidden_dim 256 --data_dir data/processed_data/hdf5_files_d_30 --feature_dim 34`

#6 `python main.py --lora_dim 16 --hidden_dim 256 --feature_dim 4`

#7 `python main.py --lora_dim 32 --hidden_dim 128`

#8 `python main.py --lora_dim 0 --hidden_dim 128 --num_layers 8`

#9 `python main.py --lora_dim 8 --hidden_dim 256`

#10 `python main.py --model_type gat --hidden_dim 512`

#11 `python main.py --lora_dim 32 --hidden_dim 256 --num_layers 8`

#12 `python main.py --lora_dim 16 --hidden_dim 384 --num_layers 8`

#13 `python main.py --lora_dim 32 --hidden_dim 128 --num_layers 8`

#14 `python main.py --model_type gat --hidden_dim 384 --num_layers 8`