# make ./autorec/data
mkdir autorec/data

# download data
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
wget https://files.grouplens.org/datasets/movielens/ml-10m.zip

# unzip file
unzip ./autorec/data/ml-1m.zip -d autorec/data/
unzip ./autorec/data/ml-10m.zip -d autorec/data/

# delete zip file
rm ml-1m.zip
rm ml-10m.zip


