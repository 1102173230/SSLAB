# Copyright (C) 2016 School of Software Engineering, Chongqing University
#
# This file is part of RecQ.
#
# RecQ is a free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
from data.rating import RatingDAO
from tool.file import FileIO
import pandas as pd
from tool.config import Config,LineConfig
from os.path import abspath
from time import strftime,localtime,time
from evaluation.measure import Measure
import numpy as np

class Recommender(object):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        self.config = conf
        self.data = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True
        self.data = RatingDAO(self.config, trainingSet, testSet)
        self.foldInfo = fold
        self.evalSettings = LineConfig(self.config['evaluation.setup'])
        self.measure = []
        self.record = []
        if self.evalSettings.contains('-cold'):
            #evaluation on cold-start users
            threshold = int(self.evalSettings['-cold'])
            removedUser = {}
            for user in self.data.testSet_u:
                if user in self.data.trainSet_u and len(self.data.trainSet_u[user])>threshold:
                    removedUser[user]=1

            for user in removedUser:
                del self.data.testSet_u[user]

            testData = []
            for item in self.data.testData:
                if item[0] not in removedUser:
                    testData.append(item)
            self.data.testData = testData

        self.num_users, self.num_items, self.train_size = self.data.trainingSize()


    def readConfiguration(self):
        self.algorName = self.config['recommender']
        self.output = LineConfig(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.ranking = LineConfig(self.config['item.ranking'])

    def printAlgorConfig(self):
        "show algorithm's configuration"
        print('Algorithm:',self.config['recommender'])
        print('Ratings dataset:',abspath(self.config['ratings']))
        if LineConfig(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:',abspath(LineConfig(self.config['evaluation.setup']).getOption('-testSet')))
        #print 'Count of the users in training set: ',len()
        print('Training set size: (user count: %d, item count %d, record count: %d)' %(self.data.trainingSize()))
        print('Test set size: (user count: %d, item count %d, record count: %d)' %(self.data.testSize()))
        print('='*80)

    def initModel(self):
        pass

    def buildModel(self):
        'build the model (for model-based algorithms )'
        pass

    def buildModel_tf(self):
        'training model on tensorflow'
        pass

    def saveModel(self):
        pass

    def loadModel(self):
        pass

    def predict(self,u,i):
        pass

    def predictForRanking(self,u):
        pass


    def checkRatingBoundary(self,prediction):
        if prediction > self.data.rScale[-1]:
            return self.data.rScale[-1]
        elif prediction < self.data.rScale[0]:
            return self.data.rScale[0]
        else:
            return round(prediction,3)

    def evalRatings(self):
        res = list() #used to contain the text of the result
        res.append('userId  itemId  original  prediction\n')
        #predict
        for ind,entry in enumerate(self.data.testData):
            user,item,rating = entry

            #predict
            prediction = self.predict(user,item)
            #denormalize
            #prediction = denormalize(prediction,self.data.rScale[-1],self.data.rScale[0])
            #####################################
            pred = self.checkRatingBoundary(prediction)
            # add prediction in order to measure
            self.data.testData[ind].append(pred)
            res.append(user+' '+item+' '+str(rating)+' '+str(pred)+'\n')
        currentTime = strftime("%Y-%m-%d %H-%M-%S",localtime(time()))
        #output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['recommender']+'@'+currentTime+'-rating-predictions'+self.foldInfo+'.txt'
            FileIO.writeFile(outDir,fileName,res)
            print('The result has been output to ',abspath(outDir),'.')
        #output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['recommender'] + '@'+currentTime +'-measure'+ self.foldInfo + '.txt'
        self.measure = Measure.ratingMeasure(self.data.testData)
        FileIO.writeFile(outDir, fileName, self.measure)
        print('The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure)))

    def evalRanking(self):
        res = []  # used to contain the text of the result

        if self.ranking.contains('-topN'):
            top = self.ranking['-topN'].split(',')
            top = [int(num) for num in top]
            N = int(top[-1])
            # if N > 100 or N < 0:
            #     print('N can not be larger than 100! It has been reassigned with 10')
            #     N = 10
            # if N > len(self.data.item):
            #     N = len(self.data.item)
        else:
            print('No correct evaluation metric is specified!')
            exit(-1)

        # res.append('userId: recommendations in (itemId, ranking score) pairs, * means the item matches.\n')
        # predict
        recList = {}
        div_recList = {}
        userN = {}
        userCount = len(self.data.testSet_u)
        #rawRes = {}
        cnt = 0
        for i, user in enumerate(self.data.testSet_u):
            cnt += 1
            print(cnt)
            itemSet = {}
            div_recList[user] = []
            # line = user + ':'
            # line = str(self.data.user[user])
            line = str(self.data.all_User[user])
            predictedItems = self.predictForRanking(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            for id, rating in enumerate(predictedItems):
                # if not self.data.rating(user, self.data.id2item[id]):
                # prediction = self.checkRatingBoundary(prediction)
                # pred = self.checkRatingBoundary(prediction)
                #####################################
                # add prediction in order to measure

                itemSet[self.data.id2item[id]] = rating

            ratedList, ratingList = self.data.userRated(user)
            for item in ratedList:
                del itemSet[item]

            Nrecommendations = []
            for item in itemSet:
                if len(Nrecommendations) < N:
                    Nrecommendations.append((item, itemSet[item]))
                else:
                    break

            Nrecommendations.sort(key=lambda d: d[1], reverse=True)
            recommendations = [item[1] for item in Nrecommendations]
            resNames = [item[0] for item in Nrecommendations]


            # find the N biggest scores
            for item in itemSet:
                ind = N
                l = 0
                r = N - 1

                if recommendations[r] < itemSet[item]:
                    while r>=l:
                        mid = int((r-l) / 2 + l)
                        if recommendations[mid] >= itemSet[item]:
                            l = mid + 1
                        elif recommendations[mid] < itemSet[item]:
                            r = mid - 1

                        if r < l:
                            ind = r
                            break
                #move the items backwards
                if ind < N - 2:
                    recommendations[ind+2:]=recommendations[ind+1:-1]
                    resNames[ind+2:]=resNames[ind+1:-1]
                if ind < N - 1:
                    recommendations[ind+1] = itemSet[item]
                    resNames[ind+1] = item

            recList[user] = list(zip(resNames, recommendations))

            if 'B_xQuAD' in self.config['recommender']:
                np_recList = np.array(recList[user])
                tmp = np_recList[:, 1].astype(np.float)
                _range = np.max(tmp) - np.min(tmp)
                tmp = (tmp - np.min(tmp)) / _range
                newList = []
                for id, item in enumerate(recList[user]):
                    newList.append([item[0], tmp[id]])
                recList[user] = newList
                
                mul1 = 1
                mul2 = 1
                reg = 11      ###############################################
                remainingList = recList[user]
                div_recList[user].append(recList[user][0])
                remainingList.remove(recList[user][0])

                while len(div_recList[user]) < 100:
                    # print(len(div_recList[user]))
                    for itemName in div_recList[user]:
                        if itemName[0] in self.data.unpopular_items:
                            mul1 = 0
                        else:
                            mul2 = 0
                    
                    # remainingList = recList[user]
                    # for item in div_recList[user]:
                    #     # print(item)
                    #     if item in remainingList:
                    #         remainingList.remove(item)
                            
                    # print(len(remainingList))
                    for id, itemName in enumerate(remainingList):
                        if itemName[0] in self.data.unpopular_items:
                            print('unpopular')
                            print(remainingList[id][1])
                            print(reg * (mul1 * (1 - self.data.preference_pop[user])))
                            remainingList[id][1] += reg * (mul1 * (1 - self.data.preference_pop[user]))
                        else:
                            # print('popular')
                            # print(remainingList[id][1])
                            # print(reg * (mul2 * self.data.preference_pop[user]))
                            # remainingList[id][1] += reg * (mul2 * self.data.preference_pop[user])
                            continue

                    remainingList.sort(key=lambda d: d[1], reverse=True)
                    if len(remainingList) != 0:
                        div_recList[user].append(remainingList[0])
                        remainingList.remove(remainingList[0])
                    else:
                        div_recList[user].append(['0', 0])
                recList[user] = div_recList[user]

            if 'S_xQuAD' in self.config['recommender']:
                np_recList = np.array(recList[user])
                tmp = np_recList[:, 1].astype(np.float)
                _range = np.max(tmp) - np.min(tmp)
                tmp = (tmp - np.min(tmp)) / _range
                newList = []
                for id, item in enumerate(recList[user]):
                    newList.append([item[0], tmp[id]])
                recList[user] = newList

                
                
                reg = 22      ###############################################
                remainingList = recList[user]
                div_recList[user].append(recList[user][0])
                remainingList.remove(recList[user][0])

                while len(div_recList[user]) < 100:
                    mul1 = 0
                    mul2 = 0
                    for itemName in div_recList[user]:
                        if itemName[0] in self.data.unpopular_items:
                            mul1 += 1
                        else:
                            mul2 += 1
                    
                    # for item in div_recList[user]:
                    #     if item in remainingList:
                    #         remainingList.remove(item)

                    for id, itemName in enumerate(remainingList):
                        if itemName[0] in self.data.unpopular_items:
                            print('unpopular')
                            print(remainingList[id][1])
                            print(reg * ((1 - mul1 / (mul1 + mul2)) * (1 - self.data.preference_pop[user])))
                            remainingList[id][1] += reg * ((1 - mul1 / (mul1 + mul2)) * (1 - self.data.preference_pop[user]))
                        else:
                            # print('popular')
                            # print(remainingList[id][1])
                            # print(reg * ((1 - mul2 / (mul1 + mul2)) * self.data.preference_pop[user]))
                            # remainingList[id][1] += reg * ((1 - mul2 / (mul1 + mul2)) * self.data.preference_pop[user])
                            continue
                    
                    remainingList.sort(key=lambda d: d[1], reverse=True)
                    if len(remainingList) != 0:
                        div_recList[user].append(remainingList[0])
                        remainingList.remove(remainingList[0])
                    else:
                        break
                        div_recList[user].append(['0', 0])

                recList[user] = div_recList[user]

            if i % 100 == 0:
                print(self.algorName, self.foldInfo, 'progress:' + str(i) + '/' + str(userCount))
            for item in recList[user]:
                # line += ' (' + item[0] + ',' + str(item[1]) + ')'
                # if item[0] in self.data.testSet_u[user]:
                #     line += '*'
                line += ' ' + str(self.data.item[item[0]])

            line += '\n'
            res.append(line)
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        if self.isOutput:
            fileName = ''
            outDir = self.output['-dir']
            fileName = self.config['recommender'] + '@' + currentTime + '-top-' + str(
            N) + 'items' + self.foldInfo + '.txt'
            FileIO.writeFile(outDir, fileName, res)
            print('The result has been output to ', abspath(outDir), '.')
        # output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['recommender'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        self.measure = Measure.rankingMeasure(self.data.testSet_u, recList, top)
        FileIO.writeFile(outDir, fileName, self.measure)
        print('The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure)))

        # res = []
        # recList = {}
        # div_recList = {}
        # userN = {}
        # userCount = len(self.data.trainSet_u)
        # for i, user in enumerate(self.data.trainSet_u):
        #     itemSet = {}
        #     div_recList[user] = []
        #     # line = user + ':'
        #     line = str(self.data.all_User[user])
        #     predictedItems = self.predictForRanking(user)
        #     # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
        #     for id, rating in enumerate(predictedItems):
        #         # if not self.data.rating(user, self.data.id2item[id]):
        #         # prediction = self.checkRatingBoundary(prediction)
        #         # pred = self.checkRatingBoundary(prediction)
        #         #####################################
        #         # add prediction in order to measure

        #         itemSet[self.data.id2item[id]] = rating

        #     ratedList, ratingList = self.data.userRated(user)
        #     for item in ratedList:
        #         del itemSet[item]

        #     Nrecommendations = []
        #     for item in itemSet:
        #         if len(Nrecommendations) < N:
        #             Nrecommendations.append((item, itemSet[item]))
        #         else:
        #             break

        #     Nrecommendations.sort(key=lambda d: d[1], reverse=True)
        #     recommendations = [item[1] for item in Nrecommendations]
        #     resNames = [item[0] for item in Nrecommendations]


        #     # find the N biggest scores
        #     for item in itemSet:
        #         ind = N
        #         l = 0
        #         r = N - 1

        #         if recommendations[r] < itemSet[item]:
        #             while r>=l:
        #                 mid = int((r-l) / 2 + l)
        #                 if recommendations[mid] >= itemSet[item]:
        #                     l = mid + 1
        #                 elif recommendations[mid] < itemSet[item]:
        #                     r = mid - 1

        #                 if r < l:
        #                     ind = r
        #                     break
        #         #move the items backwards
        #         if ind < N - 2:
        #             recommendations[ind+2:]=recommendations[ind+1:-1]
        #             resNames[ind+2:]=resNames[ind+1:-1]
        #         if ind < N - 1:
        #             recommendations[ind+1] = itemSet[item]
        #             resNames[ind+1] = item

        #     recList[user] = list(zip(resNames, recommendations))

        #     if 'B_xQuAD' in self.config['recommender']:
        #         np_recList = np.array(recList[user])
        #         tmp = np_recList[:, 1].astype(np.float)
        #         _range = np.max(tmp) - np.min(tmp)
        #         tmp = (tmp - np.min(tmp)) / _range
        #         newList = []
        #         for i, item in enumerate(recList[user]):
        #             newList.append([item[0], tmp[i]])
        #         recList[user] = newList

                
        #         mul1 = 1
        #         mul2 = 1
        #         reg = 1.5       ###############################################
        #         remainingList = recList[user]
        #         div_recList[user].append(recList[user][0])
        #         remainingList.remove(recList[user][0])

        #         while len(div_recList[user]) < 100:
        #             for itemName in div_recList[user]:
        #                 if itemName[0] in self.data.unpopular_items:
        #                     mul1 = 0
        #                 else:
        #                     mul2 = 0
                    
        #             # for item in div_recList[user]:
        #             #     if item in remainingList:
        #             #         remainingList.remove(item)

        #             for id, itemName in enumerate(remainingList):
        #                 if itemName[0] in self.data.unpopular_items:
        #                     remainingList[id][1] += reg * (mul1 * (1 - self.data.preference_pop[user]))
        #                 else:
        #                     remainingList[id][1] += reg * (mul2 * self.data.preference_pop[user])
                    
        #             remainingList.sort(key=lambda d: d[1], reverse=True)
        #             if len(remainingList) != 0:
        #                 div_recList[user].append(remainingList[0])
        #                 remainingList.remove(remainingList[0])
        #             else:
        #                 div_recList[user].append(['0', 0])
        #         recList[user] = div_recList[user]

        #     if 'S_xQuAD' in self.config['recommender']:
        #         np_recList = np.array(recList[user])
        #         tmp = np_recList[:, 1].astype(np.float)
        #         _range = np.max(tmp) - np.min(tmp)
        #         tmp = (tmp - np.min(tmp)) / _range
        #         newList = []
        #         for i, item in enumerate(recList[user]):
        #             newList.append([item[0], tmp[i]])
        #         recList[user] = newList

                
        #         mul1 = 0
        #         mul2 = 0
        #         reg = 1.5      ###############################################
        #         remainingList = recList[user]
        #         div_recList[user].append(recList[user][0])
        #         remainingList.remove(recList[user][0])

        #         while len(div_recList[user]) < 100:
        #             for itemName in div_recList[user]:
        #                 if itemName[0] in self.data.unpopular_items:
        #                     mul1 += 1
        #                 else:
        #                     mul2 += 1
                    
        #             # for item in div_recList[user]:
        #             #     if item in remainingList:
        #             #         remainingList.remove(item)

        #             for id, itemName in enumerate(remainingList):
        #                 if itemName[0] in self.data.unpopular_items:
        #                     remainingList[id][1] += reg * ((1 - mul1 / (mul1 + mul2)) * (1 - self.data.preference_pop[user]))
        #                 else:
        #                     remainingList[id][1] += reg * ((1 - mul2 / (mul1 + mul2)) * self.data.preference_pop[user])
                    
        #             remainingList.sort(key=lambda d: d[1], reverse=True)
        #             if len(remainingList) != 0:
        #                 div_recList[user].append(remainingList[0])
        #                 remainingList.remove(remainingList[0])
        #             else:
        #                 div_recList[user].append(['0', 0])

        #         recList[user] = div_recList[user]

        #     if i % 100 == 0:
        #         print(self.algorName, self.foldInfo, 'progress:' + str(i) + '/' + str(userCount))
        #     for item in recList[user]:
        #         # line += ' (' + item[0] + ',' + str(item[1]) + ')'
        #         # if item[0] in self.data.testSet_u[user]:
        #         #     line += '*'
        #         line += ' ' + str(self.data.item[item[0]])

        #     line += '\n'
        #     res.append(line)
        # currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # # output prediction result
        # if self.isOutput:
        #     fileName = ''
        #     outDir = self.output['-dir']
        #     fileName = self.config['recommender'] + '@' + currentTime + 'all-top-' + str(
        #     N) + 'items' + self.foldInfo + '.txt'
        #     FileIO.writeFile(outDir, fileName, res)
        #     print('The result has been output to ', abspath(outDir), '.')
        # # output evaluation result
        # outDir = self.output['-dir']
        # fileName = self.config['recommender'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        # self.measure = Measure.rankingMeasure(self.data.trainSet_u, recList, top)
        # FileIO.writeFile(outDir, fileName, self.measure)
        # print('The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure)))

        # with open(self.config['recommender'] + 'lastfm_rec.txt', 'w') as f:
        #     for id in range(len(self.data.trainSet_u)):
        #         relevance = self.predictForRanking(self.data.id2user[id])
        #         for s in relevance:
        #             f.write(str(s))
        #             f.write(' ')
        #         f.write('\n')
                
    def execute(self):
        self.readConfiguration()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        #load model from disk or build model
        if self.isLoadModel:
            print('Loading model %s...' %self.foldInfo)
            self.loadModel()
        else:
            print('Initializing model %s...' %self.foldInfo)
            self.initModel()
            print('Building Model %s...' %self.foldInfo)
            try:
                import tensorflow
                if self.evalSettings.contains('-tf'):
                    self.buildModel_tf()
                else:
                    self.buildModel()
            except ImportError:
                self.buildModel()

        #preict the ratings or item ranking
        print('Predicting %s...' %self.foldInfo)
        if self.ranking.isMainOn():
            self.evalRanking()
        else:
            self.evalRatings()

        #save model
        if self.isSaveModel:
            print('Saving model %s...' %self.foldInfo)
            self.saveModel()
        # with open(self.foldInfo+'measure.txt','w') as f:
        #     f.writelines(self.record)
        return self.measure



