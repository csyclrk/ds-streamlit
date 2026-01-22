# This file creates the 'pipe' NLP model and saves it as model.joblib

# Import libraries
import pandas as pd
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocessor   # utils is own library in utils.py - includes all the functions - preprocessor is the class defined - CREATED A LIBRARY

tfidf = TfidfVectorizer()
classifier = LinearSVC()

if __name__ == "__main__":
   #may need to change the following to your location of sentiments.csv
   df = pd.read_csv(r"C:\Users\casey\Desktop\Data Science\DATA\sentiments.csv") 
   pipe = make_pipeline(preprocessor(), tfidf, classifier)
   pipe.fit(df['text'],df['sentiment'])
   joblib.dump(pipe, open('model.joblib','wb'))   # dumps model into binary file - wb means open new file for purpose of writing to it in binary W - write B - binary