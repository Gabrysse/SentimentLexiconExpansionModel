
# Domain-Specific Sentiment Lexicons Induced from Labeled Documents

A.Y. 2021/2022,  Deep natural language processing project, Polytechnic of Turin


## Requirements

Verify that your virtual enviroment satisfies the requirements by running the following command in your console:

```bash
  pip install -r requirements.txt
```

## Datasets

Here you can find some useful link to download the datasets used in this repository:

- **Amazon review dataset**: https://nijianmo.github.io/amazon/index.html
- **IMDB movie review dataset**: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- **Hotel review dataset**: https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe
- **Fake and real news dataset**: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- **Coronavirus tweets NLP dataset**: https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification
- **Spam Text Message Classification dataset**: https://www.kaggle.com/datasets/team-ai/spam-text-message-classification

You should also download:
- **GloVe CommonCrawl embeddings**: https://nlp.stanford.edu/data/glove.840B.300d.zip
- **VADER lexicon**: https://raw.githubusercontent.com/cjhutto/vaderSentiment/master/vaderSentiment/vader_lexicon.txt
## Experiments

In the following section you can find how to run the different experiments.

### Domain generic

```bash
python main.py --exp d_generic
```
At the end of the training you can find the checkpoint of the model in the root folder under the name `net1.pth`.

### Domain specific

```bash
python main.py --exp d_specific --dataset Automotive.json.gz --filter_year
```

- **--dataset**: the path of Amazon review dataset you want to analyze.
- **--filter_year** _(optional)_: add this parameter if you want to select the time period May 1996 - July 2014 in the Amazon review dataset. Otherwise the time period is May 1996 - October 2018. 

At the end of the training you can find the checkpoint of the model in the root folder under the name `net2_{DATASETNAME}_{FILTERYEARVALUE}.pth`. 
`FILTERYEARVALUE` is _False_ if you did not put the argument `--filteryear` and _True_ otherwise.


### Unsupervised sentiment classification

```bash
python unsupervised_sentiment.py --checkpoint1="net2_Automotive_True.pth" --checkpoint2="net2_Automotive_False.pth" --unsup_dataset="imdb" --word_ranking

```

- **--checkpoint1**: path to model checkpoint. 
- **--checkpoint2** _(optional)_: optional additional checkpoint. If you put this parameter please put in `--checkpoint1` the one with _True_ and here the one with _False_ (like the example above).
- **--unsup_dataset**: list of dataset you want to test. Allowed values (separated by space): `"imdb hotel fake_news covid_tweet spam"`.
- **--word_ranking** _(optional)_: add this if you want to get the top 20 positive and negative for each domain in each time period (i.e. it perform the ranking for each checkpoint). If a second `--checkpoint2` is specified, then it analyze also the word that changed in positive/negative betwee the two checkpoints in input.
## Authors

- Gabriele Rosi https://github.com/Gabrysse
- Andrea Tampellini https://github.com/tampeeeeeeeeeeee

