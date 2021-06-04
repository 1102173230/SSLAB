from baseclass.IterativeRecommender import IterativeRecommender
from tool import config
import numpy as np
from random import shuffle
import tensorflow as tf

class DeepRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(DeepRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.embed_size = int(self.config['num.factors'])
        self.alpha = 0.4
        self.max_len = 50
        self.hidden_size = 128
        self.drop_rate = 0.2

        self.Q1 = tf.Variable(tf.random.truncated_normal(shape=[self.embed_size, self.hidden_size], stddev=0.005), name='Q1')
        self.Q1_bias = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_size, ], stddev=0.005), name='Q1_bias')

        self.K1 = tf.Variable(tf.random.truncated_normal(shape=[self.embed_size, self.hidden_size], stddev=0.005), name='K1')
        self.K1_bias = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_size, ], stddev=0.005), name='K1_bias')

        self.V1 = tf.Variable(tf.random.truncated_normal(shape=[self.embed_size, self.hidden_size], stddev=0.005), name='V1')
        self.V1_bias = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_size, ], stddev=0.005), name='V1_bias')

        self.linear = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_size, self.embed_size], stddev=0.005), name='linear')
        self.linear_bias = tf.Variable(tf.random.truncated_normal(shape=[self.embed_size, ], stddev=0.005), name='linear_bias')

        self.lw = tf.Variable(tf.compat.v1.random_normal(shape=[1,], stddev=0.005), tf.float32, name="lw")
        self.lb = tf.Variable(tf.compat.v1.random_normal(shape=[1,], stddev=0.005), tf.float32, name="lb")
    def readConfiguration(self):
        super(DeepRecommender, self).readConfiguration()
        # set the reduced dimension
        self.batch_size = int(self.config['batch_size'])


    def printAlgorConfig(self):
        super(DeepRecommender, self).printAlgorConfig()


    def initModel(self):
        super(DeepRecommender, self).initModel()
        self.u_idx = tf.compat.v1.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.compat.v1.placeholder(tf.int32, name="v_idx")

        self.r = tf.compat.v1.placeholder(tf.float32, name="rating")

        self.user_embeddings = tf.Variable(tf.random.truncated_normal(shape=[self.num_users, self.embed_size], stddev=0.005), name='U')
        self.item_embeddings = tf.Variable(tf.random.truncated_normal(shape=[self.num_items, self.embed_size], stddev=0.005), name='V')

        self.u_embedding = tf.nn.embedding_lookup(params=self.user_embeddings, ids=self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(params=self.item_embeddings, ids=self.v_idx)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)



    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def predictForRanking(self,u):
        'used to rank all the items for the user'
        pass

    def isConverged(self,iter):
        from math import isnan
        if isnan(self.loss):
            print('Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!')
            exit(-1)
        deltaLoss = (self.lastLoss-self.loss)
        if self.ranking.isMainOn():
            measure = self.ranking_performance()
            print('%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %s %s (Top-10 On 300 users)' \
                  %(self.algorName,self.foldInfo,iter,self.loss,deltaLoss,self.lRate, measure[-3].strip()[:11], measure[-2].strip()[:12]))
        else:
            measure = self.rating_performance()
            print('%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %5s %5s' \
                  % (self.algorName, self.foldInfo, iter, self.loss, deltaLoss, self.lRate, measure[0].strip()[:11], measure[1].strip()[:12]))
        #check if converged
        cond = abs(deltaLoss) < 1e-8
        converged = cond
        if not converged:
            self.updateLearningRate(iter)
        self.lastLoss = self.loss
        shuffle(self.data.trainingData)
        return converged

