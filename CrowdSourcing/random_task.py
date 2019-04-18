import random

class RandTask():
    """Summary of RandTask
    Getting a random index according to the given relative frequency list.

    Attributes:
        check: a boolean indicating if the RandTask has been initialized successfully
        task_num: an integer indicating number of tasks
        task_prob: an float list indicating the probability of each task
        task_thres: an float list used for comupting random task index
    """
    
    def  __init__(self, freq_lt):
        """
        Inits RandTask and check if the freq_lt is correct.
        
        Args:
            freq_lt: a float list store all tasks' relative frequency. For example:
                freq_lt = [2.0, 3.0, 1.0, 4.0] 
                It means that there are four tasks with task index 0 to 3. And the tasks' occur 
                probability are 0.2, 0.3, 0.1 and 0.4 respectively.
        """
        self.check = False        
        if type(freq_lt) != list: 
            return
        
        self.task_num = len(freq_lt)
        if(self.task_num <= 0): 
            return
    
        for i in range(self.task_num):
            if (type(freq_lt[i]) != int and type(freq_lt[i]) != float) or freq_lt[i] < 0: 
                return
                
        freq_sum = 1.0 * sum(freq_lt)
        if(freq_sum <= 0): 
            return
        
        self.task_prob = freq_lt.copy()
        self.task_thres = [0.0] * (self.task_num + 1)
        for i in range(self.task_num):
            self.task_prob[i] /= freq_sum
            self.task_thres[i + 1] = self.task_thres[i] + self.task_prob[i]
        self.task_thres[self.task_num] = 2.0
        
        self.check = True
        
        
    def get_random_task_idx(self):
        """
        Get a random task index
        
        Return:
            -1 if the check = True or
            a task index by realtive frequency
        """
        if self.check == False: 
            return -1
        
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
         
        
        
