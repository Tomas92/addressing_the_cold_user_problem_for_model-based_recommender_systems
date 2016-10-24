
# coding: utf-8

# In[1]:

from __future__ import division
import graphlab as gl
import pandas as pd
import math as math
import scipy as sci
import random
import numpy as np
import collections
import random
import csv


# In[2]:

random.seed(1)


# LOADING AND TRANSFORMING DATASET

# In[3]:

# load dataset
data = gl.SFrame.read_csv("useritemmatrix.csv",header=True,delimiter=',')

# transform dataset from graphlab SFrame to pandas DataFrame
data_pd = gl.SFrame.to_dataframe(data)


# TRAINING SET PARTITIONING

# In[4]:

# np.random.seed(1)
# data_pd_shuffled = data_pd.loc[np.random.permutation(data_pd.index)]
# n = len(data_pd_shuffled)

# partition_1 = data_pd_shuffled.iloc[0:int(math.floor(1/10*n)),]
# p1 = gl.SFrame(partition_1)
# partition_2 = data_pd_shuffled.iloc[int(math.ceil(1/10*n)):int(math.floor(2/10*n)),]
# p2 = gl.SFrame(partition_2)
# partition_3 = data_pd_shuffled.iloc[int(math.ceil(2/10*n)):int(math.floor(3/10*n)),]
# p3 = gl.SFrame(partition_3)
# partition_4 = data_pd_shuffled.iloc[int(math.ceil(3/10*n)):int(math.floor(4/10*n)),]
# p4 = gl.SFrame(partition_4)
# partition_5 = data_pd_shuffled.iloc[int(math.ceil(4/10*n)):int(math.floor(5/10*n)),]
# p5 = gl.SFrame(partition_5)
# partition_6 = data_pd_shuffled.iloc[int(math.ceil(5/10*n))+1:int(math.floor(6/10*n)),]
# p6 = gl.SFrame(partition_6)
# partition_7 = data_pd_shuffled.iloc[int(math.ceil(6/10*n)):int(math.floor(7/10*n)),]
# p7 = gl.SFrame(partition_7)
# partition_8 = data_pd_shuffled.iloc[int(math.ceil(7/10*n)):int(math.floor(8/10*n)),]
# p8 = gl.SFrame(partition_8)
# partition_9 = data_pd_shuffled.iloc[int(math.ceil(8/10*n)):int(math.floor(9/10*n)),]
# p9 = gl.SFrame(partition_9)
# partition_10 = data_pd_shuffled.iloc[int(math.ceil(9/10*n)):int(n-1),]
# p10 = gl.SFrame(partition_10)

# train_1 = pd.concat([partition_2,partition_3,partition_4,partition_5,partition_6,partition_7,partition_8,partition_9,partition_10])
# t1 = gl.SFrame(train_1)
# train_2 = pd.concat([partition_1,partition_3,partition_4,partition_5,partition_6,partition_7,partition_8,partition_9,partition_10])
# t2 = gl.SFrame(train_2)
# train_3 = pd.concat([partition_1,partition_2,partition_4,partition_5,partition_6,partition_7,partition_8,partition_9,partition_10])
# t3 = gl.SFrame(train_3)
# train_4 = pd.concat([partition_1,partition_2,partition_3,partition_5,partition_6,partition_7,partition_8,partition_9,partition_10])
# t4 = gl.SFrame(train_4)
# train_5 = pd.concat([partition_1,partition_2,partition_3,partition_4,partition_6,partition_7,partition_8,partition_9,partition_10])
# t5 = gl.SFrame(train_5)
# train_6 = pd.concat([partition_1,partition_2,partition_3,partition_4,partition_5,partition_7,partition_8,partition_9,partition_10])
# t6 = gl.SFrame(train_6)
# train_7 = pd.concat([partition_1,partition_2,partition_3,partition_4,partition_5,partition_6,partition_8,partition_9,partition_10])
# t7 = gl.SFrame(train_7)
# train_8 = pd.concat([partition_1,partition_2,partition_3,partition_4,partition_5,partition_6,partition_7,partition_9,partition_10])
# t8 = gl.SFrame(train_8)
# train_9 = pd.concat([partition_1,partition_2,partition_3,partition_4,partition_5,partition_6,partition_7,partition_8,partition_10])
# t9 = gl.SFrame(train_9)
# train_10 = pd.concat([partition_1,partition_2,partition_3,partition_4,partition_5,partition_6,partition_7,partition_8,partition_9])
# t10 = gl.SFrame(train_10)


# HYPERPARAMETER TUNING

# In[5]:

# # write hyperparamter tuning results to .csv file
# csvfile = open('parameter_tuning_results_new5.csv', 'w+')
# writer = csv.writer(csvfile, delimiter=',')
# writer.writerow(['Number of factors','Regularization parameter','Linear regularization parameter','RMSE'])

# # learn the model parameters on the training dataset
# # !!! set appropriate ranges for the hyperparameters
# # !!! perform cross-validation for tuning the hyperparameters (set different random.seeds)
# factor_index = [5,10,25,50,75,100,150,200]
# # l1_index = [1e-04,1e-05,1e-06,1e-07,1e-08,1e-09,1e-10,1e-11,1e-12]
# l2_index = [1e-04,1e-05,1e-06,1e-07,1e-08,1e-09,1e-10,1e-11,1e-12]

# # !!! perform the cross-validation here
# # split the training_data in 5 parts, use 4 parts to train and 1 part to test
# for i in factor_index:
#     for j in l1_index:
#         for h in l2_index:
#             model = gl.factorization_recommender.create(t1,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)
#             rmse1 = model.evaluate_rmse(p1,target='interaction')["rmse_overall"]
            
#             model = gl.factorization_recommender.create(t2,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)
#             rmse2 = model.evaluate_rmse(p2,target='interaction')["rmse_overall"]
            
#             model = gl.factorization_recommender.create(t3,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)
#             rmse3 = model.evaluate_rmse(p3,target='interaction')["rmse_overall"]
            
#             model = gl.factorization_recommender.create(t4,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)
#             rmse4 = model.evaluate_rmse(p4,target='interaction')["rmse_overall"]
            
#             model = gl.factorization_recommender.create(t5,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)
#             rmse5 = model.evaluate_rmse(p5,target='interaction')["rmse_overall"]
            
#             model = gl.factorization_recommender.create(t6,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)
#             rmse6 = model.evaluate_rmse(p6,target='interaction')["rmse_overall"]
            
#             model = gl.factorization_recommender.create(t7,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)
#             rmse7 = model.evaluate_rmse(p7,target='interaction')["rmse_overall"]
            
#             model = gl.factorization_recommender.create(t8,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)
#             rmse8 = model.evaluate_rmse(p8,target='interaction')["rmse_overall"]
            
#             model = gl.factorization_recommender.create(t9,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)
#             rmse9 = model.evaluate_rmse(p9,target='interaction')["rmse_overall"]
            
#             model = gl.factorization_recommender.create(t10,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)
#             rmse10 = model.evaluate_rmse(p10,target='interaction')["rmse_overall"]
            
#             rmse = (rmse1+rmse2+rmse3+rmse4+rmse5+rmse6+rmse7+rmse8+rmse9+rmse10)/10
    
#             writer.writerow([i,j,h,rmse])
            
#             print 'Computed RMSE for number of factors: ' + str(i)
#             print 'Regularization parameter: '+ str(j)
#             print 'Linear regularization parameter: ' + str(h)
            
# csvfile.close()


# RANDOMLY SELECTING COLD USERS

# In[6]:

# checking users who have many purchases
user_freq_df = pd.DataFrame.from_dict(collections.Counter(data_pd['userId']),orient='index').reset_index()
user_freq_df = user_freq_df.rename(columns={'index':'userId', 0:'freq'})

# percentage of total number of users to set as cold user
perc_cold_users = 0.25
nr_of_cold_users = int(math.floor(len(user_freq_df)*perc_cold_users))
# select the [nr_of_cold_users] with the highest number of interactions
# cold_users = user_freq_df.sort_values(by='freq',ascending=False).head(nr_of_cold_users)
cold_users = user_freq_df.sample(nr_of_cold_users,random_state=1)
cold_users = cold_users.get_value(index=range(0,(nr_of_cold_users)),col=0,takeable=True)

print 'Selecting ' + str(nr_of_cold_users) + ' cold user(s)'


# SETTINGS FOR SHOWN ITEMS (ranking lengths and item frequency threshold) AND COMPUTING THE GINI, ENTROPY AND POPENT SCORES FOR THE ITEMS

# In[7]:

# compute purchase purchase/return frequency per item
item_freq_counter = collections.Counter(data_pd['itemId'])
item_freq_df = pd.DataFrame.from_dict(item_freq_counter,orient='index').reset_index()
item_freq_df = item_freq_df.rename(columns={'index':'itemId', 0:'freq'})

# produce list of items which are at least interacted with [threshold_item] times
threshold_item = 10
threshold_item_df = item_freq_df[item_freq_df['freq']>=threshold_item]['itemId']

# GINI SCORE
# function to compute Gini
def gini(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    gini = 0.
    
    sum_probs = 0
    
    for iterator in probs:
        sum_probs += iterator * iterator

    gini = 1 - sum_probs
    return gini

unique_itemId = pd.Series(threshold_item_df)
gini_list = np.zeros(shape=(len(unique_itemId),2))
j = 0

# loop over all itemId's and compute the Gini for each item
for i in unique_itemId:
    item_i_df = data_pd[data_pd['itemId'] == i]
    gini_list[j] = [i,gini(item_i_df['interaction'])]
    j += 1

# transform to pandas DataFrame
to_df = {'itemId' : gini_list[:,0],'gini' : gini_list[:,1]}
gini_items_df = pd.DataFrame(to_df)
gini_items_df.sort_values(by='gini',inplace=True,ascending=False)

print 'Computed Gini scores for all items'

# ENTROPY SCORE
# function to compute entropy
def entropy(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    for iterator in probs:
        ent -= iterator * np.log2(iterator)

    return ent

unique_itemId = pd.Series(threshold_item_df)
entropy_list = np.zeros(shape=(len(unique_itemId),2))
j = 0

# loop over all itemId's and compute the entropy for each item
for i in unique_itemId:
    item_i_df = data_pd[data_pd['itemId'] == i]
    entropy_list[j] = [i,entropy(item_i_df['interaction'])]
    j += 1

# transform to pandas DataFrame
to_df = {'itemId' : entropy_list[:,0],'entropy' : entropy_list[:,1]}
ent_items_df = pd.DataFrame(to_df)
ent_items_df.sort_values(by='entropy',inplace=True,ascending=False)

print 'Computed entropy scores for all items'

# prepare item purchase counts for merging
item_freq_df.sort_values(by='itemId',inplace=True)
item_freq_df.set_index(keys='itemId',inplace=True)
item_freq_df['freq'] = pd.to_numeric(item_freq_df['freq'])

# POPGINI SCORE
# prepare item gini scores for merging
gini_items_df2 = gini_items_df.sort_values(by='itemId')
gini_items_df2.set_index(keys='itemId',inplace=True)

# merge frequencies and entropies
popgini_items_df = pd.concat([item_freq_df,gini_items_df2],axis=1,join='inner')

# set weights for the popgini score
weight_popularity = 0.9
weight_gini = 1
# compute popgini score
popgini_items_df['popgini'] = weight_popularity*np.log10(popgini_items_df['freq'])+weight_gini*popgini_items_df['gini']
popgini_items_df.sort_values(by='popgini',inplace=True,ascending=False)

print 'Computed PopGini scores for all items'

# POPENT SCORE
# prepare item entropies for merging
ent_items_df2 = ent_items_df.sort_values(by='itemId')
ent_items_df2.set_index(keys='itemId',inplace=True)

# merge frequencies and entropies
popent_items_df = pd.concat([item_freq_df,ent_items_df2],axis=1,join='inner')

# set weights for the popent score
weight_popularity = 0.9
weight_entropy = 1
# compute popent score
popent_items_df['popent'] = weight_popularity*np.log10(popent_items_df['freq'])+weight_entropy*popent_items_df['entropy']
popent_items_df.sort_values(by='popent',inplace=True,ascending=False)

print 'Computed PopEnt scores for all items'


# POPENT SCORE WEIGHT OPTIMIZATION

# In[59]:

# filename = 'weights_popent_final_new.csv'
# csvfile = open(filename, 'w+')
# writer = csv.writer(csvfile, delimiter=',')
# writer.writerow(['Ranking strategy','Nr. of shown items','Nr. of cold users','RMSE','Weight popularity','Weight entropy'])

# # start k and l for loop here

# weight_pop_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# weight_ent_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

# nr_of_shown_items_list = [10,100,1000,10000]

# for k in weight_pop_list:
#     for l in weight_ent_list:
        
#         # set weights for the popent score
#         weight_popularity = k
#         weight_entropy = l
#         # compute popent score
#         popent_items_df['popent'] = weight_popularity*np.log10(popent_items_df['freq'])+weight_entropy*popent_items_df['entropy']
#         popent_items_df.sort_values(by='popent',inplace=True,ascending=False)

#         print 'Computed popent scores for all items'
#         print k 
#         print l

# # start m for loop here

#         for m in nr_of_shown_items_list:
#             # set the number of items to show to the cold user
#             nr_of_shown_items = m
#             print 'Number of items shown to the cold user(s): ' + str(nr_of_shown_items)

#             # POPENT STRATEGY
#             # select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popent score
#             popent_items = popent_items_df.head(nr_of_shown_items)
#             popent_items_final = np.array(popent_items.index.values, dtype=np.int64)
#             print 'Computed ranking using popent strategy'

#             # hyperparameter ranges
#             # optimal hyperparameters
#             # num_factors
#             i = 200
#             # regularization
#             j = '1e-06'
#             # linear_regularization
#             h = '1e-07'
#             # number of shown items
#             number_of_shown_items = str(nr_of_shown_items)

#             print 'Computing results'

#             # POPENT STRATEGY
#             ranking_strategy = 'PopEnt strategy'
#             # model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
#             train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(popent_items_final))]
#             # cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
#             cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(popent_items_final))]['userId'])
#             test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(popent_items_final))]
#             train = gl.SFrame(train_pd)
#             test = gl.SFrame(test_pd)

#             model = gl.factorization_recommender.create(train,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)

#             print 'Rec sys built for popent strategy'

#             rmse = model.evaluate_rmse(test,target='interaction')["rmse_overall"]

#             print 'RMSE computed for popent strategy'

#             writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse,k,l])
            
#             print 'Finished computing results'

# csvfile.close()


# COMPOSING THE RANKINGS

# In[31]:

# set the number of items to show to the cold user
# !!! in final version, construct a loop here (and test for different number of items shown to the cold user(s))
# use 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000
# already computed: 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000
nr_of_shown_items = 50000
print 'Number of items shown to the cold user(s): ' + str(nr_of_shown_items)

# RANDOM STRATEGY
# select [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) at random
random_items = random.sample(threshold_item_df,nr_of_shown_items)
random_items = np.array(random_items,dtype='int64')
print 'Computed ranking using random strategy'

# POPULARITY STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) sorted by popularity (purchase/return frequency)
pop_items = item_freq_counter.most_common(nr_of_shown_items)
pop_items = [x[0] for x in pop_items]
pop_items = np.array(pop_items,dtype='int64')
print 'Computed ranking using popularity strategy'

# GINI STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest Gini
gini_items = gini_items_df.head(nr_of_shown_items)['itemId']
gini_items = np.array(gini_items,dtype=np.int64)
print 'Computed ranking using Gini strategy'

# ENTROPY STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest entropy
ent_items = ent_items_df.head(nr_of_shown_items)['itemId']
ent_items = np.array(ent_items,dtype=np.int64)
print 'Computed ranking using entropy strategy'

# POPGINI STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popgini score
popgini_items = popgini_items_df.head(nr_of_shown_items)
popgini_items = np.array(popgini_items.index.values, dtype=np.int64)
print 'Computed ranking using popgini strategy'

# POPENT STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popent score
popent_items = popent_items_df.head(nr_of_shown_items)
popent_items = np.array(popent_items.index.values, dtype=np.int64)
print 'Computed ranking using popent strategy'


# COMPUTING THE RESULTS FOR EACH RANKING STRATEGY

# In[32]:

# hyperparameter ranges
# optimal hyperparameters
# num_factors
i = 200
# regularization
j = '1e-06'
# linear_regularization
h = '1e-07'
# number of shown items
number_of_shown_items = str(nr_of_shown_items)

print 'Computing results'

# RANDOM STRATEGY
ranking_strategy = 'Random strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(random_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(random_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(random_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)

print 'Rec sys built for random strategy'

rmse = model.evaluate_rmse(test,target='interaction')["rmse_overall"]

print 'RMSE computed for random strategy'

filename = str('final_results_shown_items_' + number_of_shown_items + '.csv')
csvfile = open(filename, 'w+')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(['Ranking strategy','Nr. of shown items','Nr. of cold users','RMSE'])
writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse])

# POPULARITY STRATEGY
ranking_strategy = 'Popularity strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(pop_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(pop_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(pop_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)

print 'Rec sys built for popularity strategy'

rmse = model.evaluate_rmse(test,target='interaction')["rmse_overall"]

print 'RMSE computed for popularity strategy'

writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse])

# GINI STRATEGY
ranking_strategy = 'Gini strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(gini_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(gini_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(gini_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)

print 'Rec sys built for Gini strategy'

rmse = model.evaluate_rmse(test,target='interaction')["rmse_overall"]

print 'RMSE computed for Gini strategy'

writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse])

# ENTROPY STRATEGY
ranking_strategy = 'Entropy strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(ent_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(ent_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(ent_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)

print 'Rec sys built for entropy strategy'

rmse = model.evaluate_rmse(test,target='interaction')["rmse_overall"]

print 'RMSE computed for entropy strategy'

writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse])

# POPGINI STRATEGY
ranking_strategy = 'PopGini strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(popgini_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(popgini_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(popgini_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)

print 'Rec sys built for popgini strategy'

rmse = model.evaluate_rmse(test,target='interaction')["rmse_overall"]

print 'RMSE computed for popgini strategy'

writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse])

# POPENT STRATEGY
ranking_strategy = 'PopEnt strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(popent_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(popent_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(popent_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)

print 'Rec sys built for popent strategy'

rmse = model.evaluate_rmse(test,target='interaction')["rmse_overall"]

print 'RMSE computed for popent strategy'

writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse])
csvfile.close()

print 'Finished computing results'


# In[ ]:



