Project description:
  This project creates a support vector machine to make a classifier that seporates phishing URLs from Benign URLs

dataset is from kaggle at https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls and the features that come with the dataset are just the URL and a "good" or "bad" label
  
to run this code have pyhton installed and run _pip install -r requirements . txt_ to insure that the proper libraries are installed
then run _python -m src.data_ to load in the dataset. Then run src _python -m src.train_ to train the model and then _python -m src.eval_ to test and generate performance metrics

the baseline results are
  "accuracy": 0.8028491089056258,
  "precision": 0.5404957284172662,
  "recall": 0.5910571604179471,
  "f1": 0.5646468205037872

Please only use this code for defensive or research reasons only. Please refrain from using this model to help generate more believable phishing URLs.
