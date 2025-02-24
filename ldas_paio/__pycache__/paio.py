import numpy as np
import random
import statistics
import math
from collections import Counter
from sklearn.neighbors import NearestNeighbors

class PAIO:
    def __int__(self, k1=5, k2=8, rTh=5/8):
        self.k1=k1
        self.k2=k2
        self.rTh=rTh

        self.IS=[]
        self.BS=[]
        self.TS=[]

        self.nnarray = []
        self.Synthetic = []
        self.maj_index = []
        self.min_index = []

    #ded cal_nneighbors(self, N, min_sample)
    #    neigh = NearestNeighbors(n_neighbors=N).fit(min_sample)
    #    dist, indices = neigh.kneighbors(min_sample[self.min_index[0].index(ind)])

    def nearby(self, ind, min_sample, N):
        neigh = NearestNeighbors(n_neighbors=self.k2+1).fit(min_sample)
        dist, indices = neigh.kneighbors(min_sample)
        my_neigh_ind=self.min_index[0].index(indices[self.min_index[0].index(ind)])
        you_neigh_ind=[]
        for i in range(indices.shape[0]):
            if self.min_index[0].index(ind) in indices[i]:
                you_neigh_ind.append(self.min_index[0][indices[i][0]])
        s1=set(my_neigh_ind)
        s2=set(you_neigh_ind)
        s=s1.union(s2)
        gk_neig_ind=list(s)
        return gk_neig_ind

    def nbdos(self, N, M, min_sample):
        neigh = NearestNeighbors(n_neighbors=self.k2+1).fit(X)
        dist, indices = neigh.kneighbors(min_sample)
        #-----(1)ソフトコアサンプルを見つける-----
　　　　p = np.zeros(N)
        soft_core_ind=[]
        labeled_or_not={}   #key: ソフトコアサンプルのind、value: 0ならば、ラベルなし、1ならば、ラベル済み
        for i, index in enumerate(self.min_index[0]):
            neigh_label = y[indices[index, 1:self.k2 + 1]] 
            K_Nmaj = Counter(neigh_label)[1]
            p[i] = K_Nmaj / self.k2
            if p[i]>=self.rTh:
                soft_core_ind.append(index)
                labeled_or_not[index]=0
        
        #-----(2)ソフトコアサンプルの集合からランダムに1つ選ぶ-----
        clus_label=np.zeros(N)
        unlabeled_soft_core_ind=soft_core_ind
        range_k=len(soft_core_ind)
        i=0

        while sum(v==1 for v in labeled_or_not.values())<len(soft_core_ind):
            seed_ind=random.randint(0, range_k-1)
            if labeled_or_not[soft_core_ind[seed_ind]]==1:
                continue
            else:
                labeled_or_not[soft_core_ind[seed_ind]]=1
        #-----(3)現在のクラスタリングラベルを割り当てる-----
                i=i+1
                current_cl=i
                clus_label[self.min_index[0].index([soft_core_ind[seed_ind]])]=current_cl
                Li_labeled_or_not={}
                Li_labeled_or_not[soft_core_ind[seed_ind]]=0
        #-----(4)i番目のクラスタリングを拡張する-----
                while sum(v==1 for v in Li_labeled_or_not.values())<len(Li_labeled_or_not):
        #-----First-----
                    for key in Li_labeled_or_not:
                        if Li_labeled_or_not[key]==1:
                            continue
                        else:
                            if key in soft_core_ind:
                                x_i_ind=key
                                Li_labeled_or_not[x_i_ind]=1
                                gnn_ind=self.nearby(x_i_ind, min_sample, N)
                                for j in range(len(gnn_ind)):
                                    if len(gnn_ind)==0:
                                        break
                                    if gnn_ind[j] in Li_labeled_or_not.keys():
                                            continue
                                        else:
                                            Li_labeled_or_not[gnn_ind[j]]=0
        #-----Second-----
                                            clus_label[self.min_index[0].index([gnn_ind[j]])]=current_cl
                                            if gnn_ind[j] in soft_core_ind:
                                                labeled_or_not[gnn_ind[j]]=1
        clId=i        
        return clus_label, clId, soft_core_ind

    def divide_min_sample(selef, clus_label, clId, soft_core_ind, N):
        IS_ind=[]
        BS_ind=[]
        TS_ind=[]
        for i in range(N):
            if self.min_index[0][i] in soft_core_ind:
                IS_ind.append(self.min_index[0][i])
            else:
                if clus_label[i]!=0:
                    BS_ind.append(self.min_index[0][i])
                else:
                    TS_ind.append(self.min_index[0][i])
        return IS_ind, BS_ind, TS_ind

    def cal_num_syn(self, N, M):
        G=N-M
        num_syn_sample=np.zeros(N)
        for i in range(N):
            num_syn_sample[i]=math.floor(G/N)+(random.random()<=(G/N-math.floor(G/N)))
        return num_syn_sample

    def fit_sample(self, X, y):
        self.min_index = np.where(y == 0)
        self.maj_index = np.where(y == 1)

        min_sample = X[self.min_index]
        maj_sample = X[self.maj_index]

        N = len(min_sample)
        M = len(maj_sample)

        #-----divide minority class samples using NBDOS clustering-----

        self.IS, self.BS, self.TS=self.divide_min_sample((clus_label, clId, soft_core_ind=self.nbdos(N, M, min_sample)), N)

        #-----calculate the number of synthetic samples

        g=self.cal_num_syn(N, M)

        p=min_sample
        numattrs=p.shape[1]

        #-----oversampling the inland minority samples-----

        for i, index_i in enumerate(self.IS):
            L_c_label=clus_label[self.min_index[0].index(index_i)]
            CAR_i=[]
            for j, index_j in enumerate(self.min_index[0]):
                if clus_label[j]==L_c_label:
                    if index_j!=index_i:
                        CAR_i.append(index_j)
            range_l=len(CAR_i)
            count=0
            while count <g[self.min_index[0].index(index_i)]:
                s=np.zeros(numattrs)
                AR_i=CAR_i[random.randint(0, range_l-1)]
                for atti in range(numattrs):
                    gap = random.random()
                    dif = p[AR_i, atti] - p[index_i, atti]
                    s[atti] = p[index_i, atti] + gap * dif
                self.Synthetic.append(s)
                count += 1


        #-----oversampling the borderline minority samples-----

        for i, index_i in enumerate(self.BS):
            neigh = NearestNeighbors(n_neighbors=self.k1 + 1).fit(maj_sample)
            dist, indices = neigh.kneighbors(min_sample[self.min_index[0].index(index_i)])
            CAR_i=self.maj_index[0](indices[1:self.k1])
            range_l=len(CAR_i)
            count=0
            while count <g[self.min_index[0].index(index_i)]:
                s=np.zeros(numattrs)
                AR_i=CAR_i[random.randint(0, range_l-1)]
                for atti in range(numattrs):
                    gap = random.random()
                    dif = p[AR_i, atti] - p[index_i, atti]
                    s[atti] = p[index_i, atti] + gap * dif
                self.Synthetic.append(s)
                count += 1


        #-----oversampling the trapped minority samples-----

        neigh = NearestNeighbors(n_neighbors=self.k1 + 1).fit(min_sample)
        dist, indices = neigh.kneighbors(min_sample)
        dist_all=[]
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                if j==0:
                    continue
                else:
                    dist_all.append(dist[i][j])
        dTh=statistics.median(dist_all)

        for i, index_i in enumerate(self.TS):
            neigh = NearestNeighbors(n_neighbors=self.k1 + 1).fit(min_sample)
            dist, indices = neigh.kneighbors(min_sample)
            N_min_=self.min_index[0][indices[self.min_index[0].index(index_i)][1:self.k1]]
            CTh=[]
            for j, dist_j in enumerate(dist[self.min_index[0].index(index_i)][1:self.k1]):
                if dist_j<= dTh:
                    CTh.append(self.min_index[0][indices[self.min_index[0].index(index_i)][j+1]])
            s1=set(N_min)
            s2=set(CTh)
            s=s1.union(s2)
            CAR_i=list(s)
            Sw_i=np.zeros(len(CAR_i))
            for j, index_j in enumerate(CAR_i):
                dist_i_j=np.linalg.norm(X[index_i]-X[index_j])
                Sw_i[j]=((dTh/max([dist_i_j,dTh])) ** (dTh))
            sum=sum(Sw_i)
            Sp_i=np.zeros(len(CAR_i))
            for j, index_j in enumerate(CAR_i):
                Sp_i[j]=Sw[j]/sum
            Sp_sum=Sp_i
            a=0
            for j in range(len(Sp_i)):
                a=a+Sp_i[j]
                Sp_sum[j]=a

            #range_l=len(CAR_i)
            count=0
            while count <g[self.min_index[0].index(index_i)]:
                s=np.zeros(numattrs)
                r=random.random()
                for m in range(len(Sp_sum)):
                    if r<=Sp_sum[m]:
                        AR_i=CAR_i[m]

                for atti in range(numattrs):
                    gap = random.random()
                    dif = p[AR_i, atti] - p[index_i, atti]
                    s[atti] = p[index_i, atti] + gap * dif
                self.Synthetic.append(s)
                count += 1