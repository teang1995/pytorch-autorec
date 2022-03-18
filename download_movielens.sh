# make ./autorec/data
mkdir autorec/data

# download data
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
wget https://files.grouplens.org/datasets/movielens/ml-10m.zip

# unzip file
unzip ml-1m.zip -d /
unzip ml-10m.zip -d /

# delete zip file
rm ml-1m.zip
rm ml-10m.zip


