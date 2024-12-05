# PaSIG

The implementation of PaSIG in the paper:

Shiyu Wang, Gang Zhou, Jicang Lu, Jing Chen, Ningbo Huang. "Pre-trained Semantic Interaction based Inductive Graph Neural Networks for Text Classification". In 31st International Conference on Computational Linguistics (COLING 2025).

## File Trees

```
.
├── data
│     ├── embed_bert
│     ├── graph
│     ├── raw
│     │     ├── mr.labels.txt
│     │     ├── mr.texts.txt
│     │     ├── ...
│     │     ├── R52.labels.txt
│     │     └── R52.texts.txt
│     └── temp
├── log
├── out
├── proc
│     ├── dataset_config.py
│     ├── encoder_bert.py
│     ├── graph_builder.py
│     ├── my_tfidf.py
│     └── preprocess_data.py
├── ptm
│     └── bert-base-uncased
│         ├── config.json
│         ├── pytorch_model.bin
│         ├── tokenizer_config.json
│         ├── tokenizer.json
│         └── vocab.txt
└── src
    ├── bert_model.py
    ├── dataset_graph_batch.py
    ├── dataset_graph.py
    ├── dataset_text.py
    ├── gnn_layer.py
    ├── gnn_model_batch.py
    ├── gnn_model.py
    ├── train_bert.py
    ├── train_gnn_batch.py
    ├── train_gnn.py
    └── utils.py
```

## Usage

### 1. Prepare the raw data

Put the raw files in path ```data/raw/```, 
including raw text file (```*.texts.txt```) and label file 
(```*.labels.txt```).

### 2. Preprocess the data

Set dataset parameters in the file ```proc/dataset_config.py``` and 
change the selected dataset name.

Run the preprocessing code ```proc/preprocess_data.py``` and ```proc/graph_builder.py```.

```shell
  python proc/preprocess_data.py
```

```shell
  python proc/graph_builder.py
```

The intermediate data file will be saved in the path ```data/temp/``` and ```data/graph/```.

### 3. Train BERT model to obtain text embedding

- Train BERT model, run the code ```src/train_bert.py```. For example:
  
  ```shell
  python src/train_bert.py --dataset mr --epochs 20
  ```
- Get texts and words encoding by BERT, run the code ```proc/encoder_bert.py```.
  (You also need to change the ```Dataset``` name)
  
  ```shell
  python proc/encoder_bert.py
  ```

### 4. Train GNN model and observe node embedding

- Train PaSIG, run the code ```src/train_gnn.py```. For example:
  
  ```shell
  python src/train_gnn.py --dataset mr --gnn_model gfus
  ```

- Train PaSIG-S, run the code ```src/train_gnn_batch.py```. For example:
  
  ```shell
  python src/train_gnn_batch.py --dataset mr --gnn_model gfus
  ```
  
  You can choose 4 gnn components to run: ```gcn, gin, sage, gfus```.
