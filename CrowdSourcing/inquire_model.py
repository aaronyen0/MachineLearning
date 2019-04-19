class INQUIRE():
    """ 
    Implement the paper: Incremental Quality Inference in Crowdsourcing (INQUIRE)
    There are some different to the paper:
        1. adding the contributor and the contributor's rating method by statistic of confidence array
        2. the question's initial true-probability is given by the contributor's confidence
    
    Attributes:
        wkr_update_thres: a positive integer deonotes that the vote number each question need
    """
    
    def  __init__(self, wkr_update_thres = 7):
        """Initial the INQUIRE"""
        self.wkr_update_thres = wkr_update_thres

        
    class WorkerModel():
        """
        Recording datas of a worker's past performance, 
        each worker may be a voter or a contributor to some questions.
        
        Attributes:
            ID: an unique and non-negative integer represents as a specific worker
            confusion_matrix: a 2*2 matrix (list[list(float, float)]) that statistics the worker's past voting result
            gamma: a float that denotes the worker's past voting accuracy
                gamma = (confusion_mat[0][0] + confusion_mat[1][1]) / sum(confusion_mat)
            confidence_array: a list(float, flaot) that statistics the worker's past contribution
            confidence: a float that denotes the worker's past contributing accuracy
                confidence = confidence_arr[1] / sum(confidence_arr)
        """
        def  __init__(self, wkr_ID):
            self.ID = wkr_ID
            self.confusion_matrix = [[0.27, 0.23], [0.23, 0.27]]
            self.gamma = 0.54
            self.confidence_array = [0.5, 0.5]
            self.confidence = 0.5

            
    class QuestionModel():
        """
        Recording datas of a question
        
        Attributes:
            ID: an unique and non-negative integer represents as a specific question
            contributor: a class(WorkerModel) that contains informations of the question's contirbutor
            true_prob: a float that indicates the True-probability of this question's answer
            ans_lt: a list that contains informations of past answers to this question, 
                each answer to this question is recorded in a class(AnsInfo)
            qstn_result: an integer represents as the question's final result
                qstn_result = -1, default value
                qstn_result = 1, the question result is true
                qstn_result = 0, the question result is false
        """
        def  __init__(self, qstn_ID, contributor = None, true_prob = 0.5):
            self.ID = qstn_ID
            self.contributor = contributor
            self.true_prob = true_prob
            self.ans_lt = []
            self.qstn_result = -1
            
            
    class AnsInfo():
        """
        Recording informations of an answer
        
        Attributes:
            wrk_model: a class(WorkerModel) represents as a specific voter
            ans: an integer that a voter's answer to a question
                ans = 0, the voter thinked that the question's answer is false
                ans = 1, the voter thinked that the question's answer is true
        """
        def  __init__(self, wrk_model, ans):
            self.wrk_model = wrk_model
            self.ans = ans

            
    def update_qstn_model(self, qstn_model, wkr_model, ans):
        """
        Updating the question's true probability when it gets a new answer from a voter.
        Function also checks if there are enough answers to determine the question's final result.
        If the question gets enough answers, the qstn_result will be writed to 0 or 1, 
        and updating the contributor's and voters' informations related to this answer
        
        Args:
            qstn_model: a class(QuestionModel) records datas of a question
            wkr_model: a class(WorkerModel) records datas of a voter
            ans: answer from the worker to this question
        """
        qstn_model.ans_lt.append(self.AnsInfo(wkr_model, ans))
        ans_num = len(qstn_model.ans_lt)
        qstn_model.true_prob = self.prob_strategy_gamma(qstn_model.true_prob, wkr_model.gamma, ans)
        
        if qstn_model.qstn_result == -1 and ans_num >= self.wkr_update_thres:
            if qstn_model.true_prob > 0.5:
                qstn_model.qstn_result = 1
            else:
                qstn_model.qstn_result = 0
                
            if qstn_model.contributor != None:
                cfd_arr = qstn_model.contributor.confidence_array
                cfd_arr[0] += (1 - qstn_model.true_prob)
                cfd_arr[1] += qstn_model.true_prob
                qstn_model.contributor.confidence = cfd_arr[1] / (cfd_arr[0] + cfd_arr[1])
                
            ans_lt = qstn_model.ans_lt
            for j in range(ans_num):
                self.update_wkr_model(qstn_model.true_prob, ans_lt[j].wrk_model, ans_lt[j].ans)


    def update_wkr_model(self, true_prob, wkr_model, ans):
        """
        Updating a voter's information, when a question got enough answers to determine final result.
        
        Args:
            true_prob: a float that represents the probability of the question's answer is true
            wkr_model: a class(WorkerModel) records datas of the voter
            ans: answer from the voter to this question
        """
        mat = wkr_model.confusion_matrix       
        #ans = 0 or 1, updating worker_model confusion_matrix
        mat[ans][0] += (1 - true_prob)
        mat[ans][1] += true_prob
        
        # update verifier
        wkr_model.gamma = (mat[0][0] + mat[1][1]) / (mat[0][0] + mat[1][1] + mat[0][1] + mat[1][0])
        

    def prob_strategy_gamma(self, true_prob, gamma, ans):
        """
        A Probability Strategy whih is proposed in the paper(INQUIRE).
        It is a method to update the question's True-probability when it get a answer from a specific voter.
        
        Args:
            true_prob: the probability that the question's answer is true
            gamma: the voter's accuracy
            ans: answer from the worker to this question
        """
        if ans == 1:
            return (true_prob * gamma) / (true_prob * gamma + (1 - true_prob) * (1 - gamma))
        else:
            return (true_prob * (1 - gamma)) / (true_prob * (1 - gamma) + (1 - true_prob) * gamma)

'''
def test_sequence():     
    sys = INQUIRE()
    w0 = sys.WorkerModel(0)
    q1 = sys.QuestionModel(1, w0, w0.confidence)
    
    worker_lt = 100 * [None]
    for i in range(100):
        worker_lt[i] = sys.WorkerModel(i)
    
    for i in range(6):
        sys.update_qstn_model(q1, worker_lt[i], 1)
    sys.update_qstn_model(q1, worker_lt[6], 0)
    
    print(w0.confidence, w0.confidence_array, q1.true_prob)
'''    
    