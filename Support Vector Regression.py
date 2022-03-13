# -*- coding: utf-8 -*-
"""
Machine Learning
“Prediction Yelp star's” Problem


"""

#%%
# Load libraries

from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, Ridge, SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
# Load and parse the data
#Open the business.csv file with information on businesses found on Yelp

business = pd.read_csv("business.csv",low_memory=False)

#Keep only restaurants

restaurants = business[(business.categories.str.contains('Restaurants'))  | (business.categories.str.contains('Food'))]
print(restaurants.shape)
#%%
#Since we need only text, stars and mapping to business, preprocess the chunks for memory efficiency
def review_preprocess(chunk):
    chunk = chunk[['stars','text','business_id']]
    return chunk
#%%
# run time = 2-3 m
chunk_list = []  # append each chunk df here 

# Each chunk is in df format
for chunk in pd.read_csv("review.csv", chunksize=250000):  
    # perform data filtering 
    chunk_filter = review_preprocess(chunk)
    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk_filter)
    
# concat the list into dataframe 
reviews = pd.concat(chunk_list)
print(reviews)
#%%
rest_reviews = pd.merge(how='left',left=restaurants[['business_id','name']], right=reviews, left_on='business_id', right_on='business_id')
rest_reviews.dropna(inplace=True)
print(rest_reviews)
plt.hist(rest_reviews.stars)
#%%
#Keep a smaller and balanced dataset

#50000 reviews for each star

chunk_list = []

for i in range(5):
    chunk_list.append(rest_reviews[rest_reviews.stars==i+1].sample(50000))
    
small_dataset = pd.concat(chunk_list)


stars = small_dataset.stars
text = small_dataset.text

text_train, text_test, stars_train, stars_test = train_test_split(text, stars, test_size=0.2)
#%% 
# UNIGRAM - LinearSVR
# Grid Search for the Vectorizer (runtime=10 min)

param_grid_bow = {
    'Vec__min_df':[5],
    'Vec__max_df':[0.3]
}

bow_est = Pipeline([
    ('Vec',CountVectorizer()),
    ('LinearSVR',LinearSVR(max_iter=5000))
])

gs_bow = GridSearchCV(
    bow_est,param_grid_bow,
    cv=5, 
    scoring='neg_mean_squared_error'
)


gs_bow.fit(text_train,stars_train) 

print(gs_bow.best_estimator_)
pred = gs_bow.predict(text_test)
print('MSE of bow is = ',metrics.mean_squared_error(stars_test,pred))
print('R2 of bow is = ',metrics.r2_score(stars_test,pred))
#%%
# UNIGRAM TF-IDF - LinearSVR
#Grid Search without the Vectorizer

V = CountVectorizer(min_df=5,max_df=0.3).fit_transform(text)
X = TfidfTransformer().fit_transform(V)

param_grid_norm = {
    'LinearSVR__C': [0.1,1]
}

unigram_norm_est = Pipeline([
    ('Vec',CountVectorizer(min_df=5,max_df=0.3)),
    ('tfidf',TfidfTransformer()),
    ('LinearSVR',LinearSVR(max_iter=5000))
])

gs_norm = GridSearchCV(
    unigram_norm_est,param_grid_norm,
    cv=5, 
    scoring='neg_mean_squared_error'
)

gs_norm.fit(text_train,stars_train) #X or text depending if we are grid searching on the vectorizer

print(gs_norm.best_estimator_)
pred = gs_norm.predict(text_test)

print('MSE of TF-IDF unigram  = ',metrics.mean_squared_error(stars_test,pred))
print('R2 of TF-IDF unigram = ',metrics.r2_score(stars_test,pred))

#%%
# BIGRAM - LinearSVR

X = CountVectorizer(ngram_range=(1,2),min_df=7,max_df=0.2).fit_transform(text)

param_grid_bi = {
    'LinearSVR__C': [0.1,1]
}

bigram_est = Pipeline([
    ('Vec',CountVectorizer(ngram_range=(1,2),min_df=7,max_df=0.2)),
    ('LinearSVR',LinearSVR(max_iter=5000))
])

gs_bi = GridSearchCV(
    bigram_est,param_grid_bi,
    cv=5, 
    scoring='neg_mean_squared_error'
)

gs_bi.fit(text_train,stars_train)

print(gs_bi.best_estimator_)
pred = gs_bi.predict(text_test)
print('MSE of biagram  = ',metrics.mean_squared_error(stars_test,pred))
print('R2 of biagram = ',metrics.r2_score(stars_test,pred))
#%%
# BIGRAM TF-IDF - LinearSVR
#Grid Search for SGD
V = CountVectorizer(ngram_range=(1,2),min_df=7,max_df=0.2).fit_transform(text)
#V = HashingVectorizer(norm=None,ngram_range=(1,2)).fit_transform(text)
X = TfidfTransformer().fit_transform(V)

param_grid_norm = {
    'LinearSVR__C': [0.1,1]
}

bigram_norm_est = Pipeline([
    ('Vec',CountVectorizer(ngram_range=(1,2),min_df=7,max_df=0.2)),
    ('tfidf',TfidfTransformer()),
    ('LinearSVR',LinearSVR(max_iter=5000))
])

gs_bi_norm = GridSearchCV(
    bigram_norm_est,param_grid_norm,
    cv=5, 
    scoring='neg_mean_squared_error'
)

gs_bi_norm.fit(text_train,stars_train) #X or text depending if we are grid searching on the vectorizer

print(gs_norm.best_estimator_)
pred = gs_bi_norm.predict(text_test)

print('MSE of TF-IDF biagram  = ',metrics.mean_squared_error(stars_test,pred))
print('R2 of TF-IDF biagram = ',metrics.r2_score(stars_test,pred))
