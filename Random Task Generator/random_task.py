"""
使用方法
step1:rt = RandTask([3,2,4,1])
引數陣列放的是每個任務出現的相對次數(>=0)，可以為整數或浮點數
因此本例有4個task，每個task出現的機率分別為
P(task[0]) = 0.3
P(task[1]) = 0.2
P(task[2]) = 0.4
P(task[3]) = 0.1

step2：rt.get_random_task_idx()
回傳一個整數
若是-1，代表初始化時的引數不符合規則
其餘則是根據任務出現的機率抽出一個任務代號
"""
import random

class RandTask():
    def  __init__(self, freq_lt):
        self.check = False
        
        if type(freq_lt) != list: return
        
        self.task_num = len(freq_lt)
        if(self.task_num < 0): return
    
        for i in range(self.task_num):
            if (type(freq_lt[i]) != int and type(freq_lt[i]) != float) or freq_lt[i] < 0: return
                
        freq_sum = 1.0 * sum(freq_lt)
        if(freq_sum <= 0): return
        
        '''
        建立一個累積機率表，給binary search輸出taskIdx使用
        最後一格重寫為2.0一方面是蓋掉累加時的浮點誤差
        另一方面search的規則是左閉右包，但是累積機率剛好1.0時是特例[)[)[)[)[]
        若random = 1.0必須要被歸類在最後一個task，因此累積表的末欄放的數字要超過1.0
        '''
        self.task_prob = freq_lt.copy()
        self.task_thres = [0.0] * (self.task_num + 1)
        for i in range(self.task_num):
            self.task_prob[i] /= freq_sum
            self.task_thres[i + 1] = self.task_thres[i] + self.task_prob[i]
        self.task_thres[self.task_num] = 2.0
        
        self.check = True
        
    def get_random_task_idx(self):
        if self.check == False: return -1
        prob = random.uniform(0, 1)
        i = 0
        j = self.task_num
        while i + 1 < j:
            k = (i + j) // 2
            if prob < self.task_thres[k]:
                j = k
            else:
                i = k
        return i
         
        
        