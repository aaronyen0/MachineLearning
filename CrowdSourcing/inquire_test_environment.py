import inquire_model
import random_task
import numpy as np
import random


class RealWorker():
    def  __init__(self, wkr_ID, gamma, confidence):
        self.ID = wkr_ID
        self.gamma = gamma
        self.confidence = confidence

        
class RealQuestion():
    def  __init__(self, qstn_ID, result):
        self.ID = qstn_ID
        self.result = result


class InquireTestEnv():
    def  __init__(self):
        self.wkr_num = 0
        self.qstn_num = 0
        
        self.wkr_list = []
        self.real_wkr_list = []
        
        self.qstn_list = []
        self.real_qstn_list = []

        self.rand_task = random_task.RandTask([0.2, 2, 10])
        
        self.sys = inquire_model.INQUIRE(9)
        
        #Add some workers
        for i in range(20):
            self.add_worker()
            
        for i in range(100):
            self.add_question()

            
    def add_worker(self):
        #print("add_wroker")
        mu = -1.1
        sigma = 0.8
        wkr = self.sys.WorkerModel(self.wkr_num)
        self.wkr_list.append(wkr)

        prob = 1 - np.random.lognormal(mu, sigma, 2)
        prob[(prob < 0.1)] = 0.1
        prob[(prob > 0.9)] = 0.9        
        r_wkr = RealWorker(self.wkr_num, prob[0], prob[1])
        #r_wkr = RealWorker(self.wkr_num, 1.0, 0.5) 測試用
        self.real_wkr_list.append(r_wkr)        
        self.wkr_num += 1

        
    def add_question(self):
        #print("add_question")
        idx = random.randint(0, self.wkr_num - 1)
        qstn = self.sys.QuestionModel(self.qstn_num, self.wkr_list[idx], self.wkr_list[idx].confidence)
        self.qstn_list.append(qstn)
        
        if (self.real_wkr_list[idx].confidence > random.uniform(0, 1)):
            r_qstn = RealQuestion(self.qstn_num, 1)
        else:
            r_qstn = RealQuestion(self.qstn_num, 0)
        
        self.real_qstn_list.append(r_qstn)
        
        self.qstn_num += 1
        #print("qstn_num = %d, ctr = %d" %(self.qstn_num, idx))

    
    def vote_question(self):
        #print("vote_question")
        qstn_idx = -1
        for i in range(self.qstn_num):
            if self.qstn_list[i].qstn_result == -1:
                qstn_idx = i
                break
        if qstn_idx == -1:
            print("No more question")
            return;
        
        wrk_idx = random.randint(0, self.wkr_num - 1)
        
        true_false = (self.real_wkr_list[wrk_idx].gamma > random.uniform(0, 1))
        if true_false == True:
            ans = self.real_qstn_list[qstn_idx].result
        else:
            ans = 1 - self.real_qstn_list[qstn_idx].result
        
        self.sys.update_qstn_model(self.qstn_list[qstn_idx], self.wkr_list[wrk_idx], ans)
        #print("qIdx = %d, wIdx = %d, ans = %d, prob = %1.3f, real = %d" %(qstn_idx, wrk_idx, ans, self.qstn_list[qstn_idx].true_prob, self.real_qstn_list[qstn_idx].result))
 
    
    def execute_random_task(self, trials_num = 1):
        task_stat = {}
        for i in range(trials_num):
            task_idx = self.rand_task.get_random_task_idx()
            swicher = {
                0: self.add_worker,
                1: self.add_question,
                2: self.vote_question,
            }
            func = swicher.get(task_idx, lambda :print('default function')) #從map中取出方法
            func()
            
            if task_idx in task_stat:
                task_stat[task_idx] = task_stat[task_idx] + 1
            else:
                task_stat.update({task_idx: 1})
        
        for task_idx in task_stat:
            print("task: %s, cnt = %d" %(swicher.get(task_idx, 'default function'), task_stat[task_idx]))


    def show_worker_list(self, offset = 0, num = 10):
        if(offset + num > self.wkr_num):
            offset = 0
            num = self.wkr_num
        for i in range(offset, offset + num):
            print("[%3d] real: %1.3f, %1.3f,  predict: %1.3f, %1.3f" 
                  %(i, self.real_wkr_list[i].gamma, self.real_wkr_list[i].confidence, 
                    self.wkr_list[i].gamma, self.wkr_list[i].confidence))

            
    def show_question_list(self, offset = 0, num = 10):
        if(offset + num > self.qstn_num):
            offset = 0
            num = self.qstn_num
        for i in range(offset, offset + num):
            print("[%3d] real_result = %d, predict: prob = %1.3f, result = %d" %(i, self.real_qstn_list[i].result,
                  self.qstn_list[i].true_prob, self.qstn_list[i].qstn_result))
                    
       
    def test_summary(self):
        p_true = 0
        p_false = 0
        
        m_true = 0
        m_false = 0
        
        for i in range(self.qstn_num):
            if self.qstn_list[i].qstn_result != -1:
                if self.qstn_list[i].qstn_result == self.real_qstn_list[i].result:
                    p_true += 1
                else:
                    p_false += 1
                    
                if self.real_qstn_list[i].result == self.get_major_vote_ans(i):
                    m_true += 1
                else:
                    m_false += 1
                    
        print("total = %d, inquire_true_ans = %d, major_vote_true_ans = %d" 
              %(p_true + p_false, p_true, m_true))

        
    def get_major_vote_ans(self, qstn_idx):
        one_num = 0
        zero_num = 0
        length = len(self.qstn_list[qstn_idx].ans_lt)
        
        for i in range(length):
            if self.qstn_list[qstn_idx].ans_lt[i].ans == 1:
                one_num += 1
            else:
                zero_num += 1
        #print(qstn_idx, one_num, zero_num)
        if one_num > zero_num:
            return 1
        else:
            return 0
