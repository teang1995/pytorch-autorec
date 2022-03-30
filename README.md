# pytorch-autorec
Implementation of [AutoRec: Autoencoders Meet Collaborative Filtering](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)


## Directory structure
`tree -I 'autorec/data|outputs' --dirsfirst`
```
.
|-- autorec
|   |-- config
|   |   `-- ml-1m-item.yaml
|   |-- data
|   |   `-- ml-1m
|   |       |-- README
|   |       |-- movies.dat
|   |       |-- ratings.dat
|   |       `-- users.dat
|   |-- datasets
|   |   |-- __init__.py
|   |   |-- datamodule.py
|   |   `-- dataset.py
|   |-- explore
|   |   |-- explore_masked_RMSE.ipynb
|   |   `-- explore_movielens1Mdataset.ipynb
|   |-- model
|   |   |-- __init__.py
|   |   |-- autorec.py
|   |   |-- autorec_module.py
|   |   `-- loss.py
|   |-- __init__.py
|   |-- eval.py
|   |-- path.py
|   `-- train.py
|-- Dockerfile
|-- README.md
|-- requirements.txt
`-- run_container.sh
```

## Train
### 1. clone this repository
`https://github.com/teang1995/pytorch-autorec.git`

### 2. build Dockerfile
`docker build -t autorec:v1.0 .`

### 3. run bashfile for trian
`sh train.sh`