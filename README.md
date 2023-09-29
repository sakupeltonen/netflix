# Word2Vec Link Prediction Project

This project implements the Word2Vec algorithm and uses it for link prediction. The walks are generated by Node2Vec. 

## 📁 Files
- `main.py`
- `preprocess.py`: Contains preprocessing steps for netflix dataset.
- `word2vec.py`: Contains the implementation of the Word2Vec algorithm.
- `node2vec.py`: Implementation of [Node2Vec](https://github.com/aditya-grover/node2vec/tree/master) 
- `tools.py`
- `emb`: Contains saved embeddings.
- `collaborative-filtering`: Contains collaborative filtering based methods for benchmarking.

## 🔧 Usage

To run the project, use the command below. You can set various parameters related to the Word2Vec algorithm and the link prediction task via command-line arguments.

```shell
python main.py [options]
```

### 🛠️ Options
- `--data-relative-path`: Specifies the path to the input data. (Default: 'data/facebook_combined.txt')
- `--save-embedding`: If set, saves the learned embeddings.
- `--num-dimensions`: Specifies the number of dimensions for the Word2Vec model. (Default: 128)
- `--walk-length`: Specifies the maximum length of each walk. (Default: 80)
- `--num-walks`: Specifies the number of walks per source. (Default: 10)
- `--num-negative-samples`: Specifies the number of nodes used for negative sampling. (Default: 50)
- `--num-epochs`: Specifies the number of epochs in SGD. (Default: 1)
- `--learning-rate`: Specifies the learning rate in SGD. (Default: 0.01)
- `--p`: Return parameter for Node2Vec. (Default: 1)
- `--q`: Inout parameter for Node2Vec. (Default: 1)

## 🚀 Example

Run with default parameters:
```shell
python main.py
```

Run with specific parameters:
```shell
python main.py --data-relative-path data/mydata.txt --num-dimensions 64 --walk-length 40 --num-walks 20 --save-embedding
```

## 🧪 Evaluation 
TODO