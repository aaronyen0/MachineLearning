import inquire_model
import random_task
import numpy as np
import random


def inquire_test_example():
    """
    a simple example to teach you how to use the testing environent
    
    1. instantiating the InquireTestEnv()
    
    2. do something to this environment by following function:
        a. add_worker
        b. add_question
        c. vote_question
        d. execute_random_task
    
    3. check some information by following function:
        a. show_worker_list
        b. show_question_list
        c. test_summary
    """
    test_env = InquireTestEnv()
    test_env.execute_random_task(20000)
    test_env.show_question_list(0, 5)
    test_env.show_worker_list(20, 5)
    test_env.test_summary()


class RealWorker():
    """
	A worker's real information in the virtual mode.
	
	Attributes:
		ID: an unique and non-negative integer represents as a specific worker
		gamma: a float that denotes the worker's real voting accuracy
		confidence: a float that denotes the worker's contributing accuracy
    """
    def  __init__(self, wkr_ID, gamma, confidence):
        self.ID = wkr_ID
        self.gamma = gamma
        self.confidence = confidence

        
class RealQuestion():
    """
	A Question's real information in the virtual mode.
	
	Attributes:
		ID: an unique and non-negative integer represents as a specific question
		result: an integer denotes the real result of the question
			result = 0, question's real answer is false
			result = 1, question's real answer is true
    """
    def  __init__(self, qstn_ID, result):
        self.ID = qstn_ID
        self.result = result


class InquireTestEnv():
    """
	Creating a virtual environment to test the performance of the INQUIRE.
	This testing environment has some rules:
		1. Every worker may be a Voter or a Contributor.
		
		2. Each Voter has a personal fixed accuracy(gamma) to answer any question.
			I create the worker's real gamma randomly, but set model's gamma a fixed initial value a little larger than 0.5.
			
		3. Each Contributor has a personal fixed accuracy(confidence) to create a new question
			I create the worker's real confidence randomly, but set model's initial confidence a fixed initial value equals to 0.5.
			
		4. The initial True-probability of a question is equals to the contributor's model confidence at that time.
			And the question's real result is determined randomly according to the worker's real confidence.
			
		5. In my testing environment, there are only 3 external-tasks:
			a. A new worker join. New worker will get real accuracy parameters(gamma, confidence) randomly and get fixed model's parameters.
			
			b. A worker(Contributor) create a question. 
				Each question's answer is determined by contributor's real accuracy(real confidence), but the model doesn't know that. 
				We hope the model's confidence will close to real confidence after trails.
				
			c. A worker(Voter) answer a question.
				Each worker's answer is determined by voter's real accuracy(gamma), but the model doesn't know that. 
				We hope the model's gamma will close to real gamma after trails.
			
	Attributes:
		wkr_num: a non-negative integer indicates total worker number
		wkr_list: a list stores worker model class (inquire_model.WorkerModel)
		real_wkr_list: a list stores real worker class (RealWorker)
		
		qstn_num: a non-negative integer indicates total question number
		qstn_list: a list stores question model class (inquire_model.QuestionModel)
		real_qstn_list: a list stores real question class (RealQuestion)
		
		rand_task: a class(random_task) that we can get random task index after initializing
		inquire_sys: a class(INQUIRE) that contains all INQUIRE's implementation function
    """
    def  __init__(self):
        """
        Initialize InquireTestEnv.
        Using random_task.RandTask([task_0_freq, task_1_freq, task_2_freq]) to generate a random task.
        The i-th task has an occurrence probability is equals to task_i_freq / sum(all task_freq). 
        Each number of the task is denoted as follow:
            1. task_0: add a worker
            2. task_1: add a question
            3. task_2: a worker answer a question
		
        Using inquire_model.INQUIRE(thres_num) to initialize INQUIRE's method.
        The thres_number denotes that any question needs thres_num number of answers to determine final result.
        I also add some workers and some questions in the begining environment.
        """
        self.wkr_num = 0
        self.wkr_list = []
        self.real_wkr_list = []
        
        self.qstn_num = 0
        self.qstn_list = []
        self.real_qstn_list = []

        self.rand_task = random_task.RandTask([0.2, 2, 10])
        
        self.inquire_sys = inquire_model.INQUIRE(9)
        
        for i in range(100):
            self.add_worker()
            
        for i in range(100):
            self.add_question()

    
    def add_worker(self):
        """
        Adding a new worker into the environment.
        We need to update two worker list:
            1. Creating a WorkerModel and merge it into wkr_list, 
            2. Creating a RealWorker. Randomly give real accuracy parameters(gamma, confidence) to RealWorker, 
                and then merge it into real_wkr_list
        """
        mu = -1.1
        sigma = 0.8
        wkr = self.inquire_sys.WorkerModel(self.wkr_num)
        self.wkr_list.append(wkr)

        prob = 1 - np.random.lognormal(mu, sigma, 2)
        prob[(prob < 0.1)] = 0.1
        prob[(prob > 0.9)] = 0.9        
        r_wkr = RealWorker(self.wkr_num, prob[0], prob[1])
        #r_wkr = RealWorker(self.wkr_num, 1.0, 0.5) #for testing
        self.real_wkr_list.append(r_wkr)        
        self.wkr_num += 1

        
    def add_question(self):
        """
        Adding a new question into the environment.
        We need to update two question list:
            1. Creating a QuestionModel which is initialized by WorkerModel's confidence and merge it into qstn_list, 
            2. Creating a RealQuestion. Give RealQuestion a real result which is related to the contributor,
                and then merge it into real_qstn_list
        """
        idx = random.randint(0, self.wkr_num - 1)
        qstn = self.inquire_sys.QuestionModel(self.qstn_num, self.wkr_list[idx], self.wkr_list[idx].confidence)
        self.qstn_list.append(qstn)
        
        if (self.real_wkr_list[idx].confidence > random.uniform(0, 1)):
            r_qstn = RealQuestion(self.qstn_num, 1)
        else:
            r_qstn = RealQuestion(self.qstn_num, 0)
        
        self.real_qstn_list.append(r_qstn)
        
        self.qstn_num += 1
        #print("qstn_num = %d, ctr = %d" %(self.qstn_num, idx))

    
    def vote_question(self):
        """choose a non-resulted question and randomly select a worker to vote this question."""
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
        
        self.inquire_sys.update_qstn_model(self.qstn_list[qstn_idx], self.wkr_list[wrk_idx], ans)
        #print("qIdx = %d, wIdx = %d, ans = %d, prob = %1.3f, real = %d" %(qstn_idx, wrk_idx, ans, self.qstn_list[qstn_idx].true_prob, self.real_qstn_list[qstn_idx].result))
		

    def execute_random_task(self, trial_num = 1):
        """
        Randomly execute following task:
            1.add_worker
            2.add_question
            3.vote_question
		
		Args:
            trial_num: a positive integer indicates trail numbers
		"""
        task_stat = {}
        for i in range(trial_num):
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
        """
        Show worker information in the wkr_list and real_wkr_list.
		
        Args:
            offset: an non-negative integer that denotes the first worker ID you want to show
            num: a positive integer indicates numbers of workers to show
        """
        if(offset + num > self.wkr_num):
            offset = 0
            num = self.wkr_num
        for i in range(offset, offset + num):
            print("[%3d] real: %1.3f, %1.3f,  predict: %1.3f, %1.3f" 
                  %(i, self.real_wkr_list[i].gamma, self.real_wkr_list[i].confidence, 
                    self.wkr_list[i].gamma, self.wkr_list[i].confidence))

            
    def show_question_list(self, offset = 0, num = 10, show_wrong_ans = 0):
        """
        Show question information in the qstn_list and real_qstn_list.
		
        Args:
            offset: an non-negative integer that denotes the first question ID you want to show
            num: a positive integer indicates numbers of questions to show
        """
        if show_wrong_ans == 0:
            if(offset + num > self.qstn_num):
                offset = 0
                num = self.qstn_num
            for i in range(offset, offset + num):
                print("[%3d] real_result = %d, predict: prob = %1.3f, result = %d" %(i, self.real_qstn_list[i].result,
                      self.qstn_list[i].true_prob, self.qstn_list[i].qstn_result))
        else:
            for i in range(offset, self.qstn_num):
                if(self.qstn_list[i].qstn_result != self.real_qstn_list[i].result):
                    print("[%3d] real_result = %d, predict: prob = %1.3f, result = %d" %(i, self.real_qstn_list[i].result,
                          self.qstn_list[i].true_prob, self.qstn_list[i].qstn_result))
                    num -= 1
                    if(num <= 0):
                        break;
                    
                   
    def test_summary(self):
        """Show some statistic result"""
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
        """
		Get the qstn_idx-th question's result if we use the major vote.
		
		Args:
			qstn_idx: a non-negative integer denotes the index of the question in the qstn_list
			
		Return:
			1, if major vote result is True
			0, if major vote result is False
        """
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
        




