import numpy as np
import random
import statistics
import math
from utils import add_label
from collections import Counter
from sklearn.neighbors import NearestNeighbors

class PAIO:
    def __init__(self, k1=5, k2=8, rTh=5/8):
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



    def nearby(self, X, y, ind, min_sample, soft_core_ind):
        neigh = NearestNeighbors(n_neighbors=self.k2).fit(X)
        dist, indices = neigh.kneighbors(min_sample)
        my_ind=indices[self.min_index[0].index(ind)]
        my_min_ind=[]
        for i ,index_i in enumerate(my_ind):
            if y[index_i]==0:
                my_min_ind.append(my_ind[i])
        you_min_ind=[]
        for i in range(indices.shape[0]):
            if ind in indices[i]:
                if indices[i][0] in soft_core_ind:
                    you_min_ind.append(indices[i][0])
        s1=set(my_min_ind)
        s2=set(you_min_ind)
        s=s1.union(s2)
        gk_neig_ind=list(s)
        gk_neig_ind.append(ind)
        return gk_neig_ind

    def nbdos(self, X, y, N, M, min_sample):
        neigh = NearestNeighbors(n_neighbors=self.k2+1).fit(X)
        dist, indices = neigh.kneighbors(min_sample)
        #-----(1)ソフトコアサンプルを見つける-----
        soft_core_ind=[]
        labeled_or_not={}   #key: ソフトコアサンプルのind、value: 0ならば、ラベルなし、1ならば、ラベル済み
        Li_labeled_or_not={}
        for i, index in enumerate(self.min_index[0]):
            neigh_label = y[indices[i, 1:self.k2 + 1]] 
            K_Nmin = Counter(neigh_label)[0]
            if (K_Nmin / self.k2)>=self.rTh:
                soft_core_ind.append(index)
                labeled_or_not[index]=0
                Li_labeled_or_not[index]=0
        
        #-----(2)ソフトコアサンプルの集合からランダムに1つ選ぶ-----
        clus_label=np.zeros(N)
        seed_ind=0
        i=0

        while sum(v==1 for v in labeled_or_not.values())<len(soft_core_ind):
            if labeled_or_not[soft_core_ind[seed_ind]]==1:
                seed_ind=seed_ind+1
            else:
        #-----(3)現在のクラスタリングラベルを割り当てる-----
                i=i+1
                current_cl=i
                clus_label[self.min_index[0].index([soft_core_ind[seed_ind]])]=current_cl
                Li=[soft_core_ind[seed_ind]]
                num_Li_soft=1
        #-----(4)i番目のクラスタリングを拡張する-----
                for j, index_j in enumerate(soft_core_ind):
                    Li_labeled_or_not[index_j]=0
                for key in Li_labeled_or_not: 
                    if Li_labeled_or_not[key]==0:
                        if key in Li:
                            x_j_ind=key
                            gnn_ind=self.nearby(X, y, x_j_ind, min_sample, soft_core_ind)
                            for j in range(len(gnn_ind)):
                                if len(gnn_ind)==0:
                                    break
                                if gnn_ind[j] in Li:
                                    continue
                                else:
                                    Li.append(gnn_ind[j])
                                    clus_label[self.min_index[0].index([gnn_ind[j]])]=current_cl
                                    if gnn_ind[j] in soft_core_ind:
                                        num_Li_soft=num_Li_soft+1
                            Li_labeled_or_not[x_j_ind]=1
                            labeled_or_not[x_j_ind]=1
                    if sum(v==1 for v in Li_labeled_or_not.values())==num_Li_soft:
                        break
                seed_ind=seed_ind+1
        clId=i        
        return clus_label, clId, soft_core_ind

    def divide_min_sample(self, clus_label, clId, soft_core_ind, N):
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
        #print("i: {},b: {},t: {}".format(len(IS_ind),len(BS_ind),len(TS_ind)))
        return IS_ind, BS_ind, TS_ind

    def cal_num_syn(self, N, M):
        G=M-N
        num_syn_sample=np.zeros(N)
        for i in range(N):
            indicator=0
            if random.random()<=(G/N-math.floor(G/N)):
                indicator=1
            num_syn_sample[i]=math.floor(G/N)+indicator
        return num_syn_sample

    def fit_sample(self, X, y):
        min_index_np=np.where(y == 0)
        max_index_np=np.where(y == 1)
        self.min_index = list(min_index_np)
        self.maj_index = list(max_index_np)
        self.min_index[0]=list(self.min_index[0])
        self.maj_index[0]=list(self.maj_index[0])

        min_sample = X[self.min_index][0]
        maj_sample = X[self.maj_index][0]

        N = len(min_sample)
        M = len(maj_sample)

        #-----divide minority class samples using NBDOS clustering-----

        clus_label, clId, soft_core_ind=self.nbdos(X, y, N, M, min_sample)
        self.IS, self.BS, self.TS=self.divide_min_sample(clus_label, clId, soft_core_ind, N)

        #-----calculate the number of synthetic samples

        g=self.cal_num_syn(N, M)

        p=min_sample
        q=maj_sample
        numattrs=p.shape[1]

        #-----oversampling the inland minority samples-----

        for i, index_i in enumerate(self.IS):
            #print(1)
            L_c_label=clus_label[self.min_index[0].index(index_i)]
            CAR_i=[]
            for j, index_j in enumerate(self.min_index[0]):
                if clus_label[j]==L_c_label:
                    if index_j!=index_i:
                        CAR_i.append(j)
            range_l=len(CAR_i)
            if range_l==0:
                break
            count=0
            while count <g[self.min_index[0].index(index_i)]:
                s=np.zeros(numattrs)
                if range_l==1:
                    AR_i=CAR_i[0]
                else:
                    AR_i=CAR_i[random.randint(0, range_l-1)]
                for atti in range(numattrs):
                    gap = random.random()
                    dif = p[AR_i, atti] - p[self.min_index[0].index(index_i), atti]
                    s[atti] = p[self.min_index[0].index(index_i), atti] + gap * dif
                self.Synthetic.append(s)
                count += 1


        #-----oversampling the borderline minority samples-----

        for i, index_i in enumerate(self.BS):
            #print(2)
            neigh = NearestNeighbors(n_neighbors=self.k1).fit(maj_sample)
            dist, indices = neigh.kneighbors([min_sample[self.min_index[0].index(index_i)]])
            CAR_i=indices[0][0:self.k1]
            range_l=len(CAR_i)
            if range_l==0:
                break
            count=0
            while count <g[self.min_index[0].index(index_i)]:
                s=np.zeros(numattrs)
                if range_l==1:
                    AR_i=CAR_i[0]
                else:
                    AR_i=CAR_i[random.randint(0, range_l-1)]
                for atti in range(numattrs):
                    gap = random.random()
                    dif = q[AR_i, atti] - p[self.min_index[0].index(index_i), atti]
                    s[atti] = p[self.min_index[0].index(index_i), atti] + gap * dif
                self.Synthetic.append(s)
                count += 1


        #-----oversampling the trapped minority samples-----

        neigh = NearestNeighbors(n_neighbors=self.k1 + 1).fit(min_sample)
        dist, indices = neigh.kneighbors(min_sample)
        dist_all=[]
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                if dist[i][j]==0:
                    continue
                else:
                    dist_all.append(dist[i][j])
        dTh=statistics.median(dist_all)

        for i, index_i in enumerate(self.TS):
            neigh = NearestNeighbors(n_neighbors=N).fit(min_sample)
            dist, indices = neigh.kneighbors(min_sample)
            N_min=indices[self.min_index[0].index(index_i)][1:self.k1+1]
            CTh=[]
            for j, dist_j in enumerate(dist[self.min_index[0].index(index_i)][1:-1]):
                if dist_j<= dTh:
                    CTh.append(indices[self.min_index[0].index(index_i)][j+1])
                else:
                    break
            s1=set(N_min)
            s2=set(CTh)
            s=s1.union(s2)
            CAR_i=list(s)
            if len(CAR_i)==0:
                break
            Sw_i=np.zeros(len(CAR_i))
            for j, index_j in enumerate(CAR_i):
                dist_i_j=np.linalg.norm(p[self.min_index[0].index(index_i)]-p[index_j])
                Sw_i[j]=pow((dTh/max([dist_i_j,dTh])) , numattrs)
            sum1=sum(Sw_i)
            Sp_i=np.zeros(len(CAR_i))
            for j, index_j in enumerate(CAR_i):
                Sp_i[j]=Sw_i[j]/sum1
            Sp_sum=Sp_i
            a=0
            for j in range(len(Sp_i)):
                a=a+Sp_i[j]
                Sp_sum[j]=a

            count=0
            while count <g[self.min_index[0].index(index_i)]:
                s=np.zeros(numattrs)
                r=random.random()
                for m in range(len(Sp_sum)):
                    if r<=Sp_sum[m]:
                        AR_i=CAR_i[m]

                for atti in range(numattrs):
                    gap = random.random()
                    dif = p[AR_i, atti] - p[self.min_index[0].index(index_i), atti]
                    s[atti] = p[self.min_index[0].index(index_i), atti] + gap * dif
                self.Synthetic.append(s)
                count += 1
        #print(len(self.Synthetic))
        if len(self.Synthetic) > 0:
            S = np.concatenate((add_label(X, y), add_label(np.array(self.Synthetic), 0)), axis=0)
            X_new = S[:, :-1]
            y_new = S[:, -1]
            return X_new, y_new
        else:
            return X, y