# Logistic Regression Programming


**What you need to remember:**

Common steps for pre-processing a new dataset are:
- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
    ```python
    train_set.shape
    ```
    
- Reshape the datasets such that each example is now a vector of size (num_px \* num_px \* 3, 1)
    ```python
    # 注意有轉置，為了讓矩陣變成 (k * n)，n為樣本數，k為feature數
    train_set_flatten = train_set.reshape(train_set.shape[0], -1).T
    ```

- "Standardize" the data
    ```python
    train_set_x = train_set_flatten / 255.
    #train_set_x = train_set_flatten / 255.0
    ```
    
**In this exercise, you will carry out the following steps:**
- Create functions and Initialize the parameters of the model

    **1. sigmoid function**
    ```python
    def sigmoid(z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """

        s = 1 / (1 + np.exp(-z))

        return s
    ```
    
    **2. 初始化參數**
    ```python
    def initialize_with_zeros(dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)

        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """

        w = np.zeros((dim, 1))
        b = 0.0

        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))

        return w, b
    ```
    
    
    **3. 有了參數(w,b)跟X,Y，便能計算cost function 及 gradient**
    ```python
    def propagate(w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b

        Tips:
        - Write your code step by step for the propagation. np.log(), np.dot()
        """

        m = X.shape[1]

        # FORWARD PROPAGATION (FROM X TO COST)
        A = sigmoid(np.dot(w.T, X) + b)# compute activation
        cost = - np.sum(Y * np.log(A) + (1-Y) * np.log(1 - A)) / m# compute cost

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = np.dot(X, (A - Y).T) / m
        db = np.sum(A - Y) / m

        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        grads = {"dw": dw, "db": db}

        return grads, cost
    ```
    
    - Optimization
        - You have initialized your parameters.
        - You are also able to compute a cost function and its gradient.
        - Now, you want to update the parameters using gradient descent.


- Learn the parameters for the model by minimizing the cost  
    ```python
    def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for w and b.
        """
        costs = []

        for i in range(num_iterations):
            # Cost and gradient calculation (≈ 1-4 lines of code)

            ### START CODE HERE ### 
            grads, cost = propagate(w, b, X, Y)

            ### END CODE HERE ###
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]

            # update rule (≈ 2 lines of code)
            ### START CODE HERE ###
            w = w - learning_rate * dw
            b = b - learning_rate * db

            ### END CODE HERE ###
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        params = {"w": w,
                  "b": b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs
    ```




- Use the learned parameters to make predictions (on the test set)
- Analyse the results and conclude
