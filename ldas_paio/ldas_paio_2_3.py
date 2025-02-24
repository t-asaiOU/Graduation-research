import numpy as np
import random
import statistics
import math
from utils import add_label
from collections import Counter
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

class LDAS_PAIO_2_3:
    def __init__(self, k=5, k1=5, k2=8, w=0.02, rTh=5/8):
        self.k=k
        self.k1=k1
        self.k2=k2
        self.w = w
        self.rTh=rTh

        self.IS=[]
        self.BS=[]
        self.TS=[]

        self.nnarray = []
        self.Synthetic = []
        self.maj_index = []
        self.min_index = []


#----------------------------LDAS-------------------------------
    def set_dc(self, N, dist):
        K = int(np.ceil(self.w * N))
        dc = np.average(dist[:, K])
        return dc

    def cal_density(self, N, min_sample):
        neigh = NearestNeighbors(n_neighbors=N).fit(min_sample)
        dist, indices = neigh.kneighbors(min_sample)
        self.nnarray = indices
        dc = self.set_dc(N, dist)
        p = np.zeros(N)

        for i in range(N):
            for j in range(1, N):
                p[i] += np.exp(-(dist[i, j] ** 2 / dc ** 2))
        return self.normalize(p)

    def remove_Overlap_Majority(self, X, y, indices, dist, density, N):
        q = {}
        p = np.zeros(N)

        del_index_list = []
        for i, index in enumerate(self.min_index[0]):

            neigh_label = y[indices[index, 1:self.k1 + 1]]

            K_Nmaj = Counter(neigh_label)[1]
            p[i] = K_Nmaj / self.k1

            neigh_maj_index = np.where(neigh_label == 1)[0] + 1

            for j in neigh_maj_index:

                dist_to_min = dist[index, j]

                if dist_to_min == 0:
                    if indices[index, j] not in del_index_list:
                        del_index_list.append(indices[index, j])
                else:
                    overlap_value = density[i] / dist_to_min
                    if indices[index, j] in q:
                        q[indices[index, j]] += overlap_value
                    else:
                        q[indices[index, j]] = overlap_value

        if len(q)+len(del_index_list)==0:
            return X, y, len(np.where(y==1)[0])-N, p

        ol_list = list(q.values())
        Thre = np.mean(ol_list)

        for key, value in q.items():
            if value >= Thre:
                del_index_list.append(key)

        X_y = add_label(X, y)
        X_y_removed = np.delete(X_y, del_index_list, axis=0)
        X_removed, y_removed = X_y_removed[:, :-1], X_y_removed[:, -1]
        maj_temp = np.where(y_removed == 1)[0]

        need_num = len(maj_temp) - N

        return X_removed, y_removed, need_num, p

    def cal_weight(self, a, b):
        return self.normalize(a ** 2 + b ** 2)

    def cal_border_degree(self, p_list):
        border_degree_list = []
        for p in p_list:
            if p == 1 or p == 0:
                border_degree = 0
            else:
                border_degree = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            border_degree_list.append(border_degree)
        return border_degree_list

    def normalize(self, a):
        a = a.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        a = min_max_scaler.fit_transform(a)
        return a.reshape(1, -1)[0]

    def cal_num_to_gen(self, weight, G):
        sum = np.sum(weight)
        return np.rint((weight / sum) * G)

    def populate(self, i, g, p):
        N = len(p)
        if N < self.k + 1:
            range_k = N - 1
        else:
            range_k = self.k
        n = self.nnarray[i, :range_k + 1]

        numattrs = p.shape[1]
        count = 0
        while count < g[i]:
            s = np.zeros(numattrs)
            nn = random.randint(1, range_k)

            for atti in range(numattrs):
                gap = random.random()
                dif = p[n[nn], atti] - p[i, atti]
                s[atti] = p[i, atti] + gap * dif
            self.Synthetic.append(s)

            count += 1

#----------------------------PAIO-------------------------------

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
        self.min_index = np.where(y == 0)
        self.maj_index = np.where(y == 1)

        min_sample = X[self.min_index]
        maj_sample = X[self.maj_index]

        N = len(min_sample)
        M = len(maj_sample)
        #print(M)



        #-----LDAS-----

        density = self.cal_density(N, min_sample)
        neigh = NearestNeighbors(n_neighbors=self.k1 + 1).fit(X)
        dist, indices = neigh.kneighbors(X)

        X_removed, y_removed, G, p = self.remove_Overlap_Majority(X, y, indices, dist, density, N)

        border_degree_list = self.cal_border_degree(p)
        border_degree_list = np.array(border_degree_list)

        if G <= 0:
            return X_removed, y_removed
        else:

            weight = self.cal_weight(density, border_degree_list)

        #-----calculate the number of synthetic samples

        g=np.array(self.cal_num_to_gen(weight, G), dtype=int)
        #print(g)



        #-----PAIO-----
        min_index_np=np.where(y_removed == 0)
        max_index_np=np.where(y_removed == 1)
        self.min_index = list(min_index_np)
        self.maj_index = list(max_index_np)
        self.min_index[0]=list(self.min_index[0])
        self.maj_index[0]=list(self.maj_index[0])

        min_sample = X_removed[self.min_index[0]]
        maj_sample = X_removed[self.maj_index[0]]

        N = len(min_sample)
        M = len(maj_sample)
        #print(M)


        #-----divide minority class samples using NBDOS clustering-----

        clus_label, clId, soft_core_ind=self.nbdos(X_removed, y_removed, N, M, min_sample)
        self.IS, self.BS, self.TS=self.divide_min_sample(clus_label, clId, soft_core_ind, N)



        p=min_sample
        q=maj_sample
        numattrs=p.shape[1]

        #-----oversampling the inland minority samples-----

        for i, index_i in enumerate(self.IS):
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

        for i, index_i in enumerate(self.TS):
            neigh = NearestNeighbors(n_neighbors=self.k1+1).fit(min_sample)
            dist, indices = neigh.kneighbors([min_sample[self.min_index[0].index(index_i)]])
            CAR_i=indices[0][1:self.k1+1]
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


        #print(len(self.Synthetic))
        if len(self.Synthetic) > 0:
            S = np.concatenate((add_label(X_removed, y_removed), add_label(np.array(self.Synthetic), 0)), axis=0)
            X_new = S[:, :-1]
            y_new = S[:, -1]
            return X_new, y_new
        else:
            return X_removed, y_removed
