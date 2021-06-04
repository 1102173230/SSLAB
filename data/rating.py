import numpy as np
from structure import sparseMatrix,new_sparseMatrix
from tool.config import Config,LineConfig
from tool.qmath import normalize
from evaluation.dataSplit import DataSplit
import os.path
from re import split
from collections import defaultdict
class RatingDAO(object):
    'data access control'
    def __init__(self,config,trainingSet, testSet):
        self.config = config
        self.ratingConfig = LineConfig(config['ratings.setup'])
        self.user = {} #used to store the order of users in the training set
        self.item = {} #used to store the order of items in the training set
        self.id2user = {}
        self.id2item = {}
        self.all_Item = {}
        self.all_User = {}
        self.userMeans = {} #used to store the mean values of users's ratings
        self.itemMeans = {} #used to store the mean values of items's ratings
        self.globalMean = 0
        self.timestamp = {}
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict) # used to store the test set by hierarchy user:[item,rating]
        self.testSet_i = defaultdict(dict) # used to store the test set by hierarchy item:[user,rating]
        self.rScale = []
        self.popular_items = []
        self.unpopular_items = []
        self.medium_items = []

        self.popular_items_per_user = defaultdict(list)
        self.unpopular_items_per_user = defaultdict(list)

        self.Num_neighbors = 10
        self.userVector = defaultdict(list)
        self.neighbors = defaultdict(list)
        self.invert_neighbors = defaultdict(list)
        self.good_items = defaultdict(list)

        # get degree of items
        self.degree = defaultdict(lambda:0)
        self.preference_pop = defaultdict(lambda:0)
        self.pos_items_popularity = defaultdict(lambda:0)
        self.pop_trainingData = []
        self.trainingData = trainingSet[:]
        self.testData = testSet[:]

        self.__generateSet()

        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()



    def __generateSet(self):
        def cos_sim(vector_a, vector_b):
            vector_a = np.mat(vector_a)
            vector_b = np.mat(vector_b)
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            sim = 0.5 + 0.5 * cos
            return sim

        triple = []
        scale = set()
        # find the maximum rating and minimum value

        for i,entry in enumerate(self.trainingData):
            userName,itemName,rating = entry
            # makes the rating within the range [0, 1].
            #rating = normalize(float(rating), self.rScale[-1], self.rScale[0])
            #self.trainingData[i][2] = rating
            # order the user
            if userName not in self.user:
                self.user[userName] = len(self.user)
                self.id2user[self.user[userName]] = userName
            # order the item
            if itemName not in self.item:
                self.item[itemName] = len(self.item)
                self.id2item[self.item[itemName]] = itemName
                # userList.append
            self.trainSet_u[userName][itemName] = rating
            self.trainSet_i[itemName][userName] = rating
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()

        self.all_User.update(self.user)
        self.all_Item.update(self.item)
        for entry in self.testData:
            userName, itemName, rating = entry
            # order the user
            if userName not in self.user:
                self.all_User[userName] = len(self.all_User)
            # order the item
            if itemName not in self.item:
                self.all_Item[itemName] = len(self.all_Item)

            self.testSet_u[userName][itemName] = rating
            self.testSet_i[itemName][userName] = rating


        for itemName in self.trainSet_i.keys():
            self.degree[itemName] += len(self.trainSet_i[itemName])
        list_item_degrees = list(self.degree.values())
        list_item_degrees.sort()
        # print('length of list:', len(list_item_degrees))
        self.threshold_degrees = list_item_degrees[int(len(list_item_degrees) * 0.8)]   #####################
        # print('threshold: ', self.threshold_degrees)
        for itemName in self.trainSet_i.keys():
            self.pos_items_popularity[itemName] = self.degree[itemName] / self.threshold_degrees
        for userName in self.trainSet_u.keys():
            for item in self.trainSet_u[userName]:
                if self.degree[item] > self.threshold_degrees:
                    self.preference_pop[userName] += 1
            self.preference_pop[userName] /= len(self.trainSet_u[userName])

        for itemName in self.trainSet_i.keys():
            if self.degree[itemName] > self.threshold_degrees:
                self.popular_items.append(itemName)
            else:
                self.unpopular_items.append(itemName)

        for itemName in self.unpopular_items:
            if len(self.trainSet_i[itemName]) > 1:
                self.medium_items.append(itemName)

        if 'LapDQ' in self.config['recommender']:
            itemList = list(self.trainSet_i.keys())
            self.D = np.zeros((len(itemList), len(itemList)))
            for i, itemName1 in enumerate(itemList):
                print(i)
                for itemName2 in itemList[i+1: ]:
                    if (itemName1 in self.popular_items and itemName2 in self.popular_items) or (itemName1 in self.unpopular_items and itemName2 in self.unpopular_items):
                        self.D[self.item[itemName1]][self.item[itemName2]] = 1
                        self.D[self.item[itemName2]][self.item[itemName1]] = 1
            
            print('D success!')
            self.A = self.D
            self.degree_matrix = np.zeros((len(itemList), len(itemList)))
            for i in range(len(itemList)):
                self.degree_matrix[i][i] = np.sum(self.A[i])
            
            self.Ld = self.degree_matrix - self.A

        for userName in self.trainSet_u.keys():
            for itemName in self.trainSet_u[userName]:
                if itemName in self.popular_items:
                    self.popular_items_per_user[userName].append(itemName)
                else:
                    self.unpopular_items_per_user[userName].append(itemName)
    

        for i, entry in enumerate(self.trainingData):
            userName,itemName,rating = entry
            if itemName not in self.popular_items:
                continue
            self.pop_trainingData.append(entry)
    

    def __globalAverage(self):
        total = sum(self.userMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.userMeans)

    def __computeUserMean(self):
        for u in self.user:
            # n = self.row(u) > 0
            # mean = 0
            #
            # if not self.containsUser(u):  # no data about current user in training set
            #     pass
            # else:
            #     sum = float(self.row(u)[0].sum())
            #     try:
            #         mean =  sum/ n[0].sum()
            #     except ZeroDivisionError:
            #         mean = 0
            self.userMeans[u] = sum(self.trainSet_u[u].values())/float(len(self.trainSet_u[u]))

    def __computeItemMean(self):
        for c in self.item:
            self.itemMeans[c] = sum(self.trainSet_i[c].values()) / float(len(self.trainSet_i[c]))

    def getUserId(self,u):
        if u in self.user:
            return self.user[u]

    def getItemId(self,i):
        if i in self.item:
            return self.item[i]

    def trainingSize(self):
        return (len(self.user),len(self.item),len(self.trainingData))
        # ZBJ
        # return(4203, 62147, len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))
        # ZBJ
        # return(4203, 62147, len(self.testData))
    def contains(self,u,i):
        'whether user u rated item i'
        if u in self.user and i in self.trainSet_u[u]:
            return True
        else:
            return False


    def containsUser(self,u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def containsItem(self,i):
        'whether item is in training set'
        if i in self.item:
            return True
        else:
            return False

    def userRated(self,u):
        return self.trainSet_u[u].keys(),self.trainSet_u[u].values()

    def itemRated(self,i):
        return self.trainSet_i[i].keys(),self.trainSet_i[i].values()

    def row(self,u):
        k,v = self.userRated(u)
        vec = np.zeros(len(self.item))
        #print vec
        for pair in zip(k,v):
            iid = self.item[pair[0]]
            vec[iid]=pair[1]
        return vec

    def col(self,i):
        k,v = self.itemRated(i)
        vec = np.zeros(len(self.user))
        #print vec
        for pair in zip(k,v):
            uid = self.user[pair[0]]
            vec[uid]=pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user),len(self.item)))
        for u in self.user:
            k, v = self.userRated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]]=vec
        return m
    # def row(self,u):
    #     return self.trainingMatrix.row(self.getUserId(u))
    #
    # def col(self,c):
    #     return self.trainingMatrix.col(self.getItemId(c))

    def sRow(self,u):
        return self.trainSet_u[u]

    def sCol(self,c):
        return self.trainSet_i[c]

    def rating(self,u,c):
        if self.contains(u,c):
            return self.trainSet_u[u][c]
        return -1

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return len(self.trainingData)
