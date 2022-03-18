# make ./autorec/data
mkdir autorec/data

# download data
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
wget https://files.grouplens.org/datasets/movielens/ml-10m.zip

# unzip file
zip -r data/ml-1m.zip autorec/data/
zip -r data/m1-l0m.zip autorec/data/

# delete zip file
rm ml-1m.zip
rm ml-10m.zip


