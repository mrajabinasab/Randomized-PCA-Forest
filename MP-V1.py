import os
import time
import threading
import numpy as np
from progress.bar import Bar
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class RandomizedPCATree:

    def __init__(self, data, dataidx):

        self.left = None
        self.leftidx = None
        self.right = None
        self.rightidx = None
        self.data = data
        self.dataidx = dataidx
        self.size = 0
        self.rpca = None
        self.split = None
        self.npc = 5
        self.weight = [1, 1, 1, 1, 1]
        self.leftchild = None
        self.rightchild = None
        self.endcriteria = 10


    def fit(self):
        self.rpca = PCA(n_components = self.npc, svd_solver = 'randomized')
        self.rpca.fit(self.data)
        datapca = self.rpca.transform(self.data)
        projdata = np.empty(self.data.shape[0])
        for i in range(self.data.shape[0]):
            temp = 0
            for j in range(self.npc):
                temp += self.weight[j] * datapca[i, j]
            projdata[i] = temp
        mean = np.mean(projdata)
        disp = np.std(projdata)
        self.split = np.random.laplace(mean, disp, 1)
        lhc = 0
        rhc = 0
        for m in range(self.data.shape[0]):
            if projdata[m] < self.split:
                lhc += 1
            else:
                rhc += 1
        self.left = np.empty([lhc, self.data.shape[1]])
        self.leftidx = np.empty([lhc])
        self.right = np.empty([rhc, self.data.shape[1]])
        self.rightidx = np.empty([rhc])
        lhc = 0
        rhc = 0
        for z in range(self.data.shape[0]):
            if projdata[z] < self.split:
                self.left[lhc, :] = self.data[z, :]
                self.leftidx[lhc] = self.dataidx[z]
                lhc += 1
            else:
                self.right[rhc, :] = self.data[z, :]
                self.rightidx[rhc] = self.dataidx[z]
                rhc += 1
        self.leftchild = RandomizedPCATree(self.left, self.leftidx)
        self.rightchild = RandomizedPCATree(self.right, self.rightidx)
        self.size = self.dataidx.shape[0]
        if self.size > self.endcriteria:
            self.dataidx = None
        self.data = None
        if self.left.shape[0] > self.endcriteria:
            self.leftchild.fit()
        self.left = self.leftidx = None
        if self.right.shape[0] > self.endcriteria:
            self.rightchild.fit()
        self.right = self.rightidx = None
        return


def knncandidates(instance, node):
    if node.size <= node.endcriteria:
        return node.dataidx
    rpcad = node.rpca.transform(instance.reshape(1, -1))
    temp = 0
    for j in range(node.npc):
        temp += rpcad[:, j] * node.weight[j]
    if temp < node.split:
        return knncandidates(instance, node.leftchild)
    else:
        return knncandidates(instance, node.rightchild)


def forest(data, dataidx, size, bar):
    forest = []
    for i in range(size):
        forest.append(RandomizedPCATree(data, dataidx))
        forest[i].fit()
        bar.next()
    return forest


def ensembler(data, forest, bar):
    forestresult = []
    ctrl = 0
    for i in range(data.shape[0]):
        for j in forest:
            if ctrl == 0:
                result = knncandidates(data[i, :], j)
                ctrl += 1
            if ctrl != 0:
                result = np.union1d(result, knncandidates(data[i, :], j))
                ctrl += 1
            if ctrl == len(forest):
                ctrl = 0
        forestresult.append(result.astype(int))
        bar.next()
    return forestresult


def forestknn(data, forestres, k, bar):
    ensembleknn = np.empty([data.shape[0], k])
    for i in range(data.shape[0]):
        temp = np.empty(forestres[i].shape[0])
        for j in range(forestres[i].shape[0]):
            temp[j] = np.linalg.norm(data[i, :] - data[int(forestres[i][j]), :])
        sortedres = np.argsort(temp)
        for m in range(k):
            if(m < sortedres.shape[0]):
                ensembleknn[i, m] = forestres[i][sortedres[m]]
            else:
                ensembleknn[i, m] = ensembleknn[i, m-1]
        bar.next()
    return ensembleknn


def distances(data, index):
    result = np.empty(data.shape[0], dtype=float)
    for i in range(data.shape[0]):
        if i == index:
            result[i] = 0
        else:
            result[i] = np.linalg.norm(data[index] - data[i])
    return result


def knn(dists, k):
    result = np.empty((dists.shape[0], 2), dtype=float)
    res = np.empty((k, 2), dtype=float)
    for i in range(dists.shape[0]):
        result[i][0] = dists[i]
        result[i][1] = i
    sortedres = np.argsort(result[:, 0])
    for j in range(k):
        res[j][0] = dists[sortedres[j]]
        res[j][1] = sortedres[j]
    return res


def knnagg(data, k):
    result = np.empty((data.shape[0], k + 1), dtype=float)
    for i in range(data.shape[0]):
        dists = distances(data, i)
        temp = knn(dists, k)
        for j in range(k + 1):
            if j == 0:
                result[i][j] = i
            else:
                result[i][j] = temp[j - 1][1]
    return result


def missav(n, k, aggknn, knnforest, bar):
    miss = 0
    for i in range(n):
        temp = aggknn[i][1:k+1]
        for j in range(k):
            if temp[j] not in knnforest[i]:
                miss = miss + 1
        bar.next()
    return miss/(n*k)


def discrepancyav(data, k, aggknn, knnforest, bar):
    forestkdist = 0
    tablekdist = 0
    for i in range(data.shape[0]):
        forestkdist = forestkdist + np.linalg.norm(data[i, :] - data[int(knnforest[i, k-1]), :])
        tablekdist = tablekdist + np.linalg.norm(data[i, :] - data[int(aggknn[i][k]), :])
        bar.next()
    fkdav = forestkdist / data.shape[0]
    tkdav = tablekdist / data.shape[0]
    return fkdav, tkdav


def evaluate(data, k, myforest, start):
    ensbar = Bar('  Ensembling The Results...', max=data.shape[0])
    knnbar = Bar('  Finding KNNs...', max=data.shape[0])
    missbar = Bar('  Calculating Miss Rate Average...', max=data.shape[0])
    disbar = Bar('  Calculating Discrepancy Average...', max=data.shape[0])
    forestresults = ensembler(data, myforest, ensbar)
    ensbar.finish()
    myforestknn = forestknn(data, forestresults, k, knnbar)
    knnbar.finish()
    end = time.time()
    if os.path.isfile('4/knntable.csv'):
        knntable = np.loadtxt('4/knntable.csv', dtype=float, delimiter=',')
    else:
        knntable = knnagg(data, k)
        np.savetxt("4/knntable.csv", knntable, delimiter=",")
    missrateav = missav(data.shape[0], k, knntable, myforestknn, missbar)
    missbar.finish()
    estimateddav, exactdav = discrepancyav(data, k, knntable, myforestknn, disbar)
    disbar.finish()
    print("Miss Rate Average is:", missrateav)
    print("Estimated and Exact Values of Discrepancy Average Are:", estimateddav, exactdav)
    print("Running Time is:", end - start)
    

def fitforest(train, myforest, forestsize):
    fitbar = Bar('  Fitting The Trees...', max=forestsize)
    myforest.append(forest(train, np.arange(train.shape[0]), forestsize, fitbar))
    fitbar.finish()


def classify(train, testdata, forest, kc, mode, weight):
    ctrl = 0
    for j in forest:
        if ctrl == 0:
            result = knncandidates(testdata, j)
            ctrl += 1
        if ctrl != 0:
            result = np.union1d(result, knncandidates(testdata, j))
            ctrl += 1
    ensemble = np.empty(kc)
    temp = np.empty(result.shape[0])
    for i in range(result.shape[0]):
        temp[i] = np.linalg.norm(testdata - train[int(result[i])])
    sortedres = np.argsort(temp)
    for m in range(kc):
        if(m < sortedres.shape[0]):
            ensemble[m] = result[sortedres[m]]
        else:
            ensemble[m] = ensemble[m-1]    
    ctrl = 0
    for f in forest:
        if ctrl == 0:
            resultb = knncandidates(train[int(ensemble[kc-1])], f)
            ctrl += 1
        if ctrl != 0:
            resultb = np.union1d(result, knncandidates(train[int(ensemble[kc-1])], f))
            ctrl += 1
    ensembleb = np.empty(kc+1)
    temp = np.empty(resultb.shape[0])
    for p in range(resultb.shape[0]):
        temp[i] = np.linalg.norm(train[int(ensemble[kc-1])] - train[int(resultb[p])])
    sortedresb = np.argsort(temp)
    for q in range(kc+1):
        if(q < sortedresb.shape[0]):
            ensembleb[m] = resultb[sortedresb[q]]
        else:
            ensembleb[m] = ensembleb[m-1]
    if mode==1:
        indicator = np.linalg.norm(testdata - train[int(ensemble[kc-1])]) / np.linalg.norm(train[int(ensemble[kc-1])] - train[int(ensembleb[kc])])
    if mode==2:
        indicator = 0
        for u in range(kc):
            indicator = indicator + (weight[u] * (np.linalg.norm(testdata - train[int(ensemble[kc-u-1])]) / np.linalg.norm(train[int(ensemble[kc-u-1])] - train[int(ensembleb[kc-u])])))        
    return indicator
                      

def classeval(train, test, myforest, kc, mode, weight):
    labval = np.empty(test.shape[0], dtype=float)
    evalbar = Bar('  Evaluating The Model...', max=test.shape[0])
    for i in range (test.shape[0]):
        labval[i] = classify(train, test[i], myforest, kc, mode, weight)
        evalbar.next()
    evalbar.finish()
    return labval    


print("Randomized PCA Forest")
print('*'.center(80, '*'))
forestsize = 100
k = 5
data = np.loadtxt('4/full.csv', dtype=float, delimiter=',')
forest1 = []
forest2 = []
forest3 = []
forest4 = []
myforest = []
t1 = threading.Thread(target=fitforest, args=(data, forest1, forestsize // 4))
t2 = threading.Thread(target=fitforest, args=(data, forest2, forestsize // 4))
t3 = threading.Thread(target=fitforest, args=(data, forest3, forestsize // 4))
t4 = threading.Thread(target=fitforest, args=(data, forest4, forestsize // 4))
start = time.time()
t1.start()
t2.start()
t3.start()
t4.start()
t1.join()
t2.join()
t3.join()
t4.join()
myforest = forest1[0] + forest2[0] + forest3[0] + forest4[0]
evaluate(data, k, myforest, start)