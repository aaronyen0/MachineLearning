# Logistic Regression Programming


**What you need to remember:**

## Common steps for pre-processing a new dataset are:

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
    
## In this exercise, you will carry out the following steps:

- **Create functions and Initialize the parameters of the model**

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


- **Learn the parameters for the model by minimizing the cost**

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
            grads, cost = propagate(w, b, X, Y)

            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]

            # update rule (≈ 2 lines of code)
            w = w - learning_rate * dw
            b = b - learning_rate * db

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



- **Use the learned parameters to make predictions (on the test set)**

    **1. GRADED FUNCTION: predict**
    
    ```python
    def predict(w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''

        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)

        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        A = sigmoid(np.dot(w.T, X) + b)
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        Y_prediction = (A > 0.5) + np.zeros((1,m))
        """
        for i in range(A.shape[1]):
            if A[0, i] > 0.5:
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0
        """

        assert(Y_prediction.shape == (1, m))

        return Y_prediction
    ```
    
    
    **2. GRADED FUNCTION: predict**
    
    ```python
    def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
        """
        Builds the logistic regression model by calling the function you've implemented previously

        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations

        Returns:
        d -- dictionary containing information about the model.
        """

        # initialize parameters with zeros (≈ 1 line of code)
        w, b = initialize_with_zeros(X_train.shape[0])

        # Gradient descent (≈ 1 line of code)
        parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

        # Retrieve parameters w and b from dictionary "parameters"
        w = parameters["w"]
        b = parameters["b"]

        # Predict test/train set examples (≈ 2 lines of code)
        Y_prediction_test = predict(w, b, X_test)
        Y_prediction_train = predict(w, b, X_train)

        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : w, 
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}

        return d    
    ```
    
## Analyse the results and conclude

- **Example of a picture that was wrongly classified.**
    
    ```python
    index = 1
    # 記得要把flatten後的矩陣轉回RPG格式 height * width * channel
    plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
    print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")
    ```
    
- **Plot learning curve (with costs)**
    ```python
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()
    ```
    
    ![](https://i.imgur.com/r6mxgTL.gif)
    
    You can see the cost decreasing. It shows that the parameters are being learned. However, you see that you could train the model even more on the training set. Try to increase the number of iterations in the cell above and rerun the cells. You might see that the training set accuracy goes up, but the test set accuracy goes down. This is called overfitting.
    
    
- **Plot accuracy curve**

    ```python
    train_accuracy = []
    test_accuracy = []
    iterations = []
    step = 50

    w, b = initialize_with_zeros(train_set_x.shape[0])

    for i in range(0, 4000, step):
        parameters, grads, costs = optimize(w, b, train_set_x, train_set_y, step, learning_rate = 0.005, print_cost = False)
        w = parameters["w"]
        b = parameters["b"]

        # Predict test/train set examples (≈ 2 lines of code)
        Y_prediction_test = predict(w, b, test_set_x)
        Y_prediction_train = predict(w, b, train_set_x)

        train_accuracy.append(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100)
        test_accuracy.append(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100)
        iterations.append(i + step)

    plt.plot(iterations, train_accuracy, 'r-')
    plt.plot(iterations, test_accuracy, 'b-')

    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title("Red:TrainAccuracy, Blue:TestAccuracy")
    plt.show()   
    ```
    ![](https://i.imgur.com/oXgDjrT.gif)


## Further analysis (optional/ungraded exercise)

- **Choice of learning rate**

    In order for Gradient Descent to work you must choose the learning rate wisely. The learning rate  αα  determines how rapidly we update the parameters. If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.
     
    ```python
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    
    """ Result
    learning rate is: 0.01
    train accuracy: 99.52153110047847 %
    test accuracy: 68.0 %

    -------------------------------------------------------

    learning rate is: 0.001
    train accuracy: 88.99521531100478 %
    test accuracy: 64.0 %

    -------------------------------------------------------

    learning rate is: 0.0001
    train accuracy: 68.42105263157895 %
    test accuracy: 36.0 %

    -------------------------------------------------------
    """
    ```
    
    ![](https://i.imgur.com/J8QUoOm.gif)

    - **Interpretation:**

        1. Different learning rates give different costs and thus different predictions results.

        2. If the learning rate is too large (0.01), the cost may oscillate up and down. It may even diverge (though in this example, using 0.01 still eventually ends up at a good value for the cost).

        3. A lower cost doesn't mean a better model. You have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.

        4. In deep learning, we usually recommend that you:

        5. Choose the learning rate that better minimizes the cost function.

        6. If your model overfits, use other techniques to reduce overfitting. (We'll talk about this in later videos.)


- **Test with your own image (optional/ungraded exercise)**

    - **We preprocess the image to fit your algorithm.**
    
    ```python
    # 設定檔名路徑
    my_image = "my_image.jpg"   # change this to the name of your image file 
    fname = "images/" + my_image
    
    # 讀檔案
    image = np.array(ndimage.imread(fname, flatten=False))
    
    # 重點：將原始檔案resize成模型可以讀的大小
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    
    #放入模型中
    my_predicted_image = predict(d["w"], d["b"], my_image)
    ```
    


## What to remember from this assignment:

- Preprocessing the dataset is important.

- You implemented each function separately: initialize(), propagate(), optimize(). Then you built a model().

- Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference to the algorithm. You will see more examples of this later in this course!


- Try and play with include:
 
    - Play with the learning rate and the number of iterations

    - Try different initialization methods and compare the results

    - Test other preprocessings (center the data, or divide each row by its standard deviation)



<br><br><br><br><br><br><br><br><br><br><br><br>
