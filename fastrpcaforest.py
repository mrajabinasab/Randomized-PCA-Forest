import threading
import numpy as np
import argparse
import sys
from progress.bar import Bar
from sklearn.decomposition import PCA


class RandomizedPCATree:

    def __init__(self, data, dataidx, npc, ec):

        self.left = None
        self.leftidx = None
        self.right = None
        self.rightidx = None
        self.data = data
        self.dataidx = dataidx
        self.size = 0
        self.npc = npc
        self.split = None
        self.leftchild = None
        self.rightchild = None
        self.endcriteria = ec


    def fit(self):
        self.split = np.empty(self.npc)
        for u in range(self.npc):
            tempmean = np.mean(self.data[:, u])
            tempdisp = np.std(self.data[:, u])
            self.split[u] = np.random.laplace(tempmean, tempdisp, 1)
        lhc = 0
        rhc = 0
        for m in range(self.data.shape[0]):
            lc = rc = 0
            for n in range(self.npc):
                if self.data[m, n] < self.split[n]:
                    lc += 1
                else:
                    rc += 1
            if rc < lc:
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
            lc = rc = 0
            for n in range(self.npc):
                if self.data[z, n] < self.split[n]:
                    lc += 1
                else:
                    rc += 1
            if rc < lc:
                self.left[lhc, :] = self.data[z, :]
                self.leftidx[lhc] = self.dataidx[z]
                lhc += 1
            else:
                self.right[rhc, :] = self.data[z, :]
                self.rightidx[rhc] = self.dataidx[z]
                rhc += 1
        self.leftchild = RandomizedPCATree(self.left, self.leftidx, self.npc, self.endcriteria)
        self.rightchild = RandomizedPCATree(self.right, self.rightidx, self.npc, self.endcriteria)
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
    lc = rc = 0
    for j in range(node.npc):
        if instance[j] < node.split[j]:
            lc += 1
        else:
            rc += 1
    if rc < lc:
        return knncandidates(instance, node.leftchild)
    else:
        return knncandidates(instance, node.rightchild)


def ensembler(data, forest, verbos):
    if verbos:
        bar = Bar('  Ensembling The Results...', max=data.shape[0])
    forestresult = []
    for i in range(data.shape[0]):
        ctrl = 0
        for j in forest:
            if ctrl == 0:
                result = knncandidates(data[i], j)
                ctrl = 1
                continue
            if ctrl != 0:
                result = np.union1d(result, knncandidates(data[i], j))
        forestresult.append(result.astype(int))
        if verbos:
            bar.next()
    if verbos:
        bar.finish()
    return forestresult


def forest(data, dataidx, size, npc, ec, verbos):
    if verbos:
        bar = Bar('  Fitting The Trees...', max=size)
    forest = []
    for i in range(size):
        forest.append(RandomizedPCATree(data, dataidx, npc, ec))
        forest[i].fit()
        if verbos:
            bar.next()
    if verbos:
        bar.finish()
    return forest


def forestknn(data, forestres, k, verbos):
    if verbos:
        bar = Bar('  Finding kNNs...', max=data.shape[0])
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
        if verbos:
            bar.next()
    if verbos:
        bar.finish()
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


def knnagg(data, k, verbos):
    if verbos:
        bar = Bar('  Constructing kNN Table...', max=data.shape[0])
    result = np.empty((data.shape[0], k + 1), dtype=float)
    for i in range(data.shape[0]):
        dists = distances(data, i)
        temp = knn(dists, k)
        for j in range(k + 1):
            if j == 0:
                result[i][j] = i
            else:
                result[i][j] = temp[j - 1][1]
        if verbos: 
            bar.next()
    if verbos:
        bar.finish()
    return result


def recall(data, k, aggknn, knnforest):
    miss = 0
    for i in range(data.shape[0]):
        temp = aggknn[i][1:k+1]
        for j in range(k):
            if temp[j] not in knnforest[i]:
                miss = miss + 1
    return (1-(miss/(data.shape[0]*k)))


def discrepancyav(data, k, aggknn, knnforest):
    forestkdist = 0
    tablekdist = 0
    for i in range(data.shape[0]):
        forestkdist = forestkdist + np.linalg.norm(data[i, :] - data[int(knnforest[i, k-1]), :])
        tablekdist = tablekdist + np.linalg.norm(data[i, :] - data[int(aggknn[i][k]), :])
    fkdav = forestkdist / data.shape[0]
    tkdav = tablekdist / data.shape[0]
    return tkdav / fkdav


def fitforest(train, myforest, forestsize, npc, ec, verbos):
    myforest.append(forest(train, np.arange(train.shape[0]), forestsize, npc, ec, verbos))


def evaluate(data, knntable, k, p, l, f, t, v):
    if t > f:
        t = f
    rpca = PCA(n_components = p, svd_solver = 'randomized')
    rpca.fit(data)
    projdata = rpca.transform(data)
    forests = [[] for _ in range(t)]
    myforest = []
    base_size = f // t
    remainder =  f % t
    threads = []
    for i in range(t):
        size = base_size + (1 if i < remainder else 0)  
        thread = threading.Thread(target=fitforest, args=(projdata, forests[i], size, p, l, v))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    myforest = sum((forest[0] for forest in forests), [])
    approxknn = forestknn(data, ensembler(projdata, myforest, v), k, v)
    return recall(data, k, knntable, approxknn), discrepancyav(data, k, knntable, approxknn)


parser = argparse.ArgumentParser(description='FAST Randomized PCA Forest')
parser.add_argument("-d", "--dataset", help="Path to the dataset csv files.", default="./data.csv", type = str)
parser.add_argument("-k", "--k", help="Value of k.", default=5, type = int)
parser.add_argument("-p", "--principalcomponents", help="Number of principal components to use.", default=5, type = int)
parser.add_argument("-l", "--leafsize", help="Maximum size of a node to be considered a leaf.", default=10, type = int)
parser.add_argument("-f", "--forestsize", help="Number of trees in the forest.", default=40, type = int)
parser.add_argument("-t", "--threads", help="Number of threads to use.", default=4, type = int)
parser.add_argument("-r", "--recursionlimit", help="Maximum number of recursions allowed.", default=1000, type = int)
parser.add_argument("-v", "--verbos", help="Set it to 1 to enable verbosity, 0 to disable it.", default=1, type = int)
args = parser.parse_args()
sys.setrecursionlimit(args.recursionlimit)
print("FAST Randomized PCA Forest")
print('*'.center(80, '*'))
data = np.loadtxt(args.dataset, dtype=float, delimiter=',')
rpca = PCA(n_components = args.principalcomponents, svd_solver = 'randomized')
rpca.fit(data)
projdata = rpca.transform(data)
if args.threads > args.forestsize:
    args.threads = args.forestsize
forests = [[] for _ in range(args.threads)]
myforest = []
base_size = args.forestsize // args.threads
remainder =  args.forestsize % args.threads
threads = []
for i in range(args.threads):
    size = base_size + (1 if i < remainder else 0)  
    thread = threading.Thread(target=fitforest, args=(projdata, forests[i], size, args.principalcomponents, args.leafsize, args.verbos))
    threads.append(thread)
    thread.start()
for thread in threads:
    thread.join()
myforest = sum((forest[0] for forest in forests), [])
approxknn = forestknn(data, ensembler(projdata, myforest, args.verbos), args.k, args.verbos)
knntable = knnagg(data, args.k, args.verbos) 
print("Recall is: ", recall(data, args.k, knntable, approxknn))
print("Average Discrepancy Ratio is: ", discrepancyav(data, args.k, knntable, approxknn))

