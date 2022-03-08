# pytorch-autorec
Implementation of [AutoRec: Autoencoders Meet Collaborative Filtering](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)


## Directory structure
```
.
|-- config
|   `-- train_config.yaml
|-- data
|   |-- movielens-10M
|   |   |-- README.html
|   |   |-- allbut.pl
|   |   |-- movies.dat
|   |   |-- ratings.dat
|   |   |-- split_ratings.sh
|   |   `-- tags.dat
|   `-- movielens-1M
|       |-- README
|       |-- movies.dat
|       |-- ratings.dat
|       `-- users.dat
|-- datasets
|   |-- __init__.py
|   |-- datamodule.py
|   `-- dataset.py
|-- explore
|-- model
|   |-- __init__.py
|   |-- autorec.py
|   |-- autorrec_module.py
|   `-- loss.py
|-- Dockerfile
|-- README.md
|-- eval.py
|-- path.py
|-- requirements.txt
`-- train.py
```