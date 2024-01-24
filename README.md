# MA4DIV
This is the code of MA4DIV which is a reinforcement learning approach for search result diversification. The framework of this project is based on https://github.com/hijkzzz/pymarl2.

## Run an experiment
```
python main.py --config=qmix --env-config=search_engine
```

## data case
The indexes of queries and documents in the DU-DIV are shown in:
```
data/baidu/query_doc_subtopics_doc15_2.json
```
We will release the whole DU-DIV dataset in the future. The whole dataset includes the features of all 4473 queries and 67095 documents with $1024-dims$. 
