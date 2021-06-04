#coding:utf8
from baseclass.IterativeRecommender import IterativeRecommender
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
import tensorflow as tf
import numpy as np
import math
import os
import matplotlib.pyplot as plt

class self_FBPR_LN(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(self_FBPR_LN, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(self_FBPR_LN, self).initModel()


    def buildModel(self):

        print('Preparing item sets...')
        self.PositiveSet = defaultdict(dict)
        for user in self.data.user:
            for item in self.data.trainSet_u[user]:
                if self.data.trainSet_u[user][item] >= 1:
                    self.PositiveSet[user][item] = 1

        print('training...')
        iteration = 0
        itemList = self.data.item.keys()
        while iteration < self.maxIter:
            self.loss = 0
            for user in self.PositiveSet:
                u = self.data.user[user]
                for item in self.PositiveSet[user]:
                    i = self.data.item[item]

                    item_j = choice(itemList)
                    while (self.PositiveSet[user].has_key(item_j)):
                        item_j = choice(itemList)
                    j = self.data.item[item_j]
                    self.optimization(u,i,j)

            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break

    def optimization(self,u,i,j):
        s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
        self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
        self.Q[i] += self.lRate * (1 - s) * self.P[u]
        self.Q[j] -= self.lRate * (1 - s) * self.P[u]

        self.P[u] -= self.lRate * self.regU * self.P[u]
        self.Q[i] -= self.lRate * self.regI * self.Q[i]
        self.Q[j] -= self.lRate * self.regI * self.Q[j]
        self.loss += -log(s)

    def predict(self,user,item):

        if self.data.containsUser(user) and self.data.containsItem(item):
            u = self.data.getUserId(user)
            i = self.data.getItemId(item)
            predictRating = sigmoid(self.Q[i].dot(self.P[u]))
            return predictRating
        else:
            return sigmoid(self.data.globalMean)

    def next_batch(self):
        batch_id=0
        while batch_id<self.train_size:
            if batch_id+self.batch_size<=self.train_size:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id,self.batch_size+batch_id)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id,self.batch_size+batch_id)]
                batch_id+=self.batch_size
            else:
                users = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                items = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id=self.train_size

            preference_pop = []
            popularity = []
            for user in users:
                preference_pop.append(self.data.preference_pop[user])
            for item in items:
                tmp = min((self.data.pos_items_popularity[item]), 2)# min(1.0 / self.data.pos_items_popularity[item], 10)
                popularity.append(tmp)
            
            u_idx, i_idx, j_idx = [], [], []
            pop_item_idx, unpop_item_idx = [], []
            neg_unpop_item_idx = []
            neg_pop_item_idx = []
            pop_mask_flag = []
            unpop_mask_flag = []
            item_list = self.data.item.keys()
            for i, user in enumerate(users):
                u_idx.append(self.data.user[user])
                i_idx.append(self.data.item[items[i]])
                if items[i] in self.data.popular_items_per_user[user]:
                    neg_item = choice(self.data.popular_items)
                    while neg_item in self.data.trainSet_u[user]:
                        neg_item = choice(self.data.popular_items)
                    j_idx.append(self.data.item[neg_item])
                else:
                    neg_item = choice(self.data.unpopular_items)
                    while neg_item in self.data.trainSet_u[user]:
                        neg_item = choice(self.data.unpopular_items)
                    j_idx.append(self.data.item[neg_item])

##############################################################################################
                tmp = []
                mask_tmp = []
                if len(self.data.popular_items_per_user[user]) != 0: 
                    if len(self.data.popular_items_per_user[user]) > self.max_len:
                        samples = np.random.choice(np.array(self.data.popular_items_per_user[user]), self.max_len, replace=False)
                        for sample in list(samples):
                            tmp.append(self.data.item[sample])
                            mask_tmp.append(1.)
                    else:
                        for pop_item in self.data.popular_items_per_user[user]: 
                            tmp.append(self.data.item[pop_item])
                            mask_tmp.append(1.)
                            if len(tmp) >= self.max_len:
                                break
                while len(tmp) < self.max_len:
                    tmp.append(0)
                    mask_tmp.append(-1e-8)  
                pop_item_idx.append(tmp[::-1])
                pop_mask_flag.append(mask_tmp[::-1])
                
                tmp = []
                mask_tmp = []
                if len(self.data.unpopular_items_per_user[user]) != 0:
                    if(len(self.data.unpopular_items_per_user[user])) > self.max_len:
                        samples = np.random.choice(np.array(self.data.unpopular_items_per_user[user]), self.max_len, replace=False)
                        for sample in list(samples):
                            tmp.append(self.data.item[sample])
                            mask_tmp.append(1.)
                    else:
                        for unpop_item in self.data.unpopular_items_per_user[user]:
                            tmp.append(self.data.item[unpop_item])
                            mask_tmp.append(1.)
                            if len(tmp) >= self.max_len:
                                break
                while len(tmp) < self.max_len:
                    tmp.append(0)
                    mask_tmp.append(-1e-8) 
                unpop_item_idx.append(tmp[::-1])
                unpop_mask_flag.append(mask_tmp[::-1])

                tmp = []
                if len(self.data.popular_items_per_user[user]) != 0:
                    for _ in self.data.popular_items_per_user[user]:
                        neg_pop_item = choice(self.data.popular_items)
                        while neg_pop_item in self.data.popular_items_per_user[user]:
                            neg_pop_item = choice(self.data.popular_items)
                        tmp.append(self.data.item[neg_pop_item])
                        if len(tmp) >= self.max_len:
                            break
                while len(tmp) < self.max_len:
                    tmp.append(0)
                neg_pop_item_idx.append(tmp[::-1])

                tmp = []
                if len(self.data.unpopular_items_per_user[user]) != 0:
                    for _ in self.data.unpopular_items_per_user[user]:
                        neg_unpop_item = choice(self.data.unpopular_items)
                        while neg_unpop_item in self.data.unpopular_items_per_user[user]:
                            neg_unpop_item = choice(self.data.unpopular_items)
                        tmp.append(self.data.item[neg_unpop_item])
                        if len(tmp) >= self.max_len:
                            break
                while len(tmp) < self.max_len:
                    tmp.append(0)
                neg_unpop_item_idx.append(tmp[::-1])
                
###############################################################################################
                
                
            yield u_idx,i_idx,j_idx, preference_pop, popularity, pop_item_idx, unpop_item_idx, neg_pop_item_idx, neg_unpop_item_idx, pop_mask_flag, unpop_mask_flag


    def buildModel_tf(self):
        super(self_FBPR_LN, self).buildModel_tf()
        def attention(query, key, value):
            "Compute 'Scaled Dot Product Attention'"
            d_k = query.shape[-1]
            key_ = tf.transpose(key, (0, 2, 1))

            scores = tf.matmul(query, key_) / (self.hidden_size**0.5)
            p_attn = tf.nn.softmax(scores, axis=-1)
            
            return tf.matmul(p_attn, value), p_attn

        def layernorm(input, epsilon = 1e-12, max = 1000):
            """ Layer normalizes a 2D tensor along its second axis, which corresponds to batch """
            print(input)
            m, v = tf.nn.moments(input, [1], keepdims=True)
            normalised_input = (input - m) / tf.sqrt(v + epsilon)
            return normalised_input * self.lw + self.lb

        self.neg_idx = tf.compat.v1.placeholder(tf.int32, name="neg_holder")
        self.pop_item_idx = tf.compat.v1.placeholder(tf.int32, name="pop_holder")
        self.unpop_item_idx = tf.compat.v1.placeholder(tf.int32, name="unpop_holder")
        self.neg_unpop_item_idx = tf.compat.v1.placeholder(tf.int32, name="neg_unpop_holder")
        self.neg_pop_item_idx = tf.compat.v1.placeholder(tf.int32, name="neg_pop_holder")
        self.pop_mask_flag = tf.compat.v1.placeholder(tf.float32, name="pop_mask_holder")
        self.unpop_mask_flag = tf.compat.v1.placeholder(tf.float32, name="unpop_mask_holder")

        self.neg_item_embedding = tf.nn.embedding_lookup(params=self.V, ids=self.neg_idx)
        self.pop_item_embedding = tf.nn.embedding_lookup(params=self.V, ids=self.pop_item_idx)
        self.unpop_item_embedding = tf.nn.embedding_lookup(params=self.V, ids=self.unpop_item_idx)
        self.neg_pop_item_embedding = tf.nn.embedding_lookup(params=self.V, ids=self.neg_pop_item_idx)
        self.neg_unpop_item_embedding = tf.nn.embedding_lookup(params=self.V, ids=self.neg_unpop_item_idx)

        
        self.masked_pop_sequence = tf.expand_dims(self.pop_mask_flag, -1) * self.pop_item_embedding
        self.masked_unpop_sequence = tf.expand_dims(self.unpop_mask_flag, -1) * self.unpop_item_embedding
        self.neg_masked_unpop_sequence = tf.expand_dims(self.unpop_mask_flag, -1) * self.neg_unpop_item_embedding
        self.neg_masked_pop_sequence = tf.expand_dims(self.pop_mask_flag, -1) * self.neg_pop_item_embedding


        self.preference_pop = tf.compat.v1.placeholder(tf.float32)
        self.popularity = tf.compat.v1.placeholder(tf.float32)

        query1 = tf.matmul(self.masked_pop_sequence, self.Q1) + self.Q1_bias
        key1 = tf.matmul(self.masked_pop_sequence, self.K1) + self.K1_bias
        value1 = tf.matmul(self.masked_pop_sequence, self.V1) + self.V1_bias

        pop_output, _ = attention(query1, key1, value1)
        pop_output = tf.matmul(pop_output, self.linear) + self.linear_bias
        pop_output = tf.nn.dropout(pop_output, rate=self.drop_rate)
        pop_output = layernorm(tf.add(pop_output, self.masked_pop_sequence))

        query1_neg = tf.matmul(self.neg_masked_pop_sequence, self.Q1) + self.Q1_bias
        key1_neg = tf.matmul(self.neg_masked_pop_sequence, self.K1) + self.K1_bias
        value1_neg = tf.matmul(self.neg_masked_pop_sequence, self.V1) + self.V1_bias

        pop_output_neg, _ = attention(query1_neg, key1_neg, value1_neg)
        pop_output_neg = tf.matmul(pop_output_neg, self.linear) + self.linear_bias
        pop_output_neg = tf.nn.dropout(pop_output_neg, rate=self.drop_rate)
        pop_output_neg = layernorm(tf.add(pop_output_neg, self.neg_masked_pop_sequence))

        query2 = tf.matmul(self.masked_unpop_sequence, self.Q1) + self.Q1_bias
        key2 = tf.matmul(self.masked_unpop_sequence, self.K1) + self.K1_bias
        value2 = tf.matmul(self.masked_unpop_sequence, self.V1) + self.V1_bias

        unpop_output, _ = attention(query2, key2, value2)
        unpop_output = tf.matmul(unpop_output, self.linear) + self.linear_bias
        unpop_output = tf.nn.dropout(unpop_output, rate=self.drop_rate)
        unpop_output = layernorm(tf.add(unpop_output, self.masked_unpop_sequence))

        query2_neg = tf.matmul(self.neg_masked_unpop_sequence, self.Q1) + self.Q1_bias
        key2_neg = tf.matmul(self.neg_masked_unpop_sequence, self.K1) + self.K1_bias
        value2_neg = tf.matmul(self.neg_masked_unpop_sequence, self.V1) + self.V1_bias

        unpop_output_neg, _ = attention(query2_neg, key2_neg, value2_neg)
        unpop_output_neg = tf.matmul(unpop_output_neg, self.linear) + self.linear_bias
        unpop_output_neg = tf.nn.dropout(unpop_output_neg, rate=self.drop_rate)
        unpop_output_neg = layernorm(tf.add(unpop_output_neg, self.neg_masked_unpop_sequence))

        z = tf.math.log(tf.sigmoid(tf.reduce_sum(input_tensor=tf.multiply(pop_output[:,-1,:], unpop_output[:,-1,:]), axis=1)) + 1e-8)\
                                 + tf.math.log(1 - tf.sigmoid(tf.reduce_sum(input_tensor=tf.multiply(pop_output[:,-1,:], unpop_output_neg[:,-1,:]), axis=1)) + 1e-8)
        z2 = tf.math.log(tf.sigmoid(tf.reduce_sum(input_tensor=tf.multiply(pop_output[:,-1,:], unpop_output[:,-1,:]), axis=1)) + 1e-8)\
                                 + tf.math.log(1 - tf.sigmoid(tf.reduce_sum(input_tensor=tf.multiply(pop_output_neg[:,-1,:], unpop_output[:,-1,:]), axis=1)) + 1e-8)

        y = tf.reduce_sum(input_tensor=tf.multiply(self.user_embedding, self.item_embedding),axis=1)\
                                 -tf.reduce_sum(input_tensor=tf.multiply(self.user_embedding,self.neg_item_embedding),axis=1)

        loss = -tf.reduce_sum(input_tensor=tf.math.log(tf.sigmoid(y))) - self.alpha * (tf.reduce_sum(input_tensor=z) + tf.reduce_sum(input_tensor=z2)) + self.regU * (tf.nn.l2_loss(self.user_embedding) +
                                                                       tf.nn.l2_loss(self.item_embedding) +
                                                                       tf.nn.l2_loss(self.neg_item_embedding)+
                                                                       tf.nn.l2_loss(self.unpop_item_embedding)+
                                                                       tf.nn.l2_loss(self.pop_item_embedding)+
                                                                       tf.nn.l2_loss(self.Q1)+
                                                                       tf.nn.l2_loss(self.V1)+
                                                                       tf.nn.l2_loss(self.K1)+
                                                                       tf.nn.l2_loss(self.Q1_bias)+
                                                                       tf.nn.l2_loss(self.K1_bias)+
                                                                       tf.nn.l2_loss(self.V1_bias)+
                                                                       tf.nn.l2_loss(self.linear)+
                                                                       tf.nn.l2_loss(self.linear_bias))

        opt = tf.compat.v1.train.AdamOptimizer(self.lRate)

        train = opt.minimize(loss)

        with tf.compat.v1.Session() as sess:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            all_loss = []
            for iteration in range(self.maxIter):
                epoch_loss = []
                for n,batch in enumerate(self.next_batch()):
                    user_idx, i_idx, j_idx, preference_pop, popularity, pop_item_idx, unpop_item_idx, neg_pop_item_idx, neg_unpop_item_idx, pop_mask_flag, unpop_mask_flag = batch
                    _, l = sess.run([train, loss], feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx,self.v_idx: i_idx, self.preference_pop: preference_pop, self.popularity: popularity, self.pop_item_idx: pop_item_idx, self.unpop_item_idx: unpop_item_idx, self.neg_pop_item_idx: neg_pop_item_idx, self.neg_unpop_item_idx: neg_unpop_item_idx, self.pop_mask_flag:pop_mask_flag, self.unpop_mask_flag:unpop_mask_flag})
                    print('training:', iteration + 1, 'batch', n, 'loss:', l)
                    epoch_loss.append(l)
                all_loss.append(sum(epoch_loss) / len(epoch_loss))
                print('avg_loss:', sum(epoch_loss) / len(epoch_loss))
                self.P,self.Q = sess.run([self.U,self.V])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.Q.dot(self.P[u])
        else:
            return [self.data.globalMean] * self.num_items


