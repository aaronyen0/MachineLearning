# Building Deep Neural Network Program 1

![](https://i.imgur.com/rOatR2K.png)

## 1 - Packages

- numpy: is the main package for scientific computing with Python.
- matplotlib: is a library to plot graphs in Python.
- np.random.seed(1): is used to keep all the random function calls consistent. It will help us grade your work. Please don't change the seed.


## 2 - Process

- Initialize the parameters for a two-layer network and for an **L-layer** neural network.
- Implement the forward propagation module (shown in purple in the figure below).
    - Complete the LINEAR part of a layer's forward propagation step (resulting in Z).
    - We give you the ACTIVATION function (relu/sigmoid).
    - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
    - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer L). This gives you a new L_model_forward function.
- Compute the loss.
- Implement the backward propagation module (denoted in red in the figure below).
    - Complete the LINEAR part of a layer's backward propagation step.
    - We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward)
    - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
    - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
- Finally update the parameters.

    ![](https://i.imgur.com/iHMAGBo.png)

## 3 - Initialization

### 3.1 - L-layer Neural Network - Initialize parameters

- initialize_parameters_deep(layer_dims):
    ```python
    def initialize_parameters_deep(layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)            # number of layers in the network

        for l in range(1, L):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            ### END CODE HERE ###

            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters
    ```
    
    - 回傳的parameters是一個dictionary，使用方法例如：
        - W1 = parameters['W1']
        - b1 = parameters['b1']
    - 注意迴圈中的L是layer_dims的長度，這個長度有包含input layer，所以實際上總共只有L-1組參數，range(1,L)，個數就是L-1組。


## 4 - Forward propagation module

- Forward propagation有兩組公式交互使用：

    - Linear Forward：
        ![](https://i.imgur.com/gjx8lvE.png)

    - Linear-Activation Forward：
        ![](https://i.imgur.com/qQi0hCS.png)
    
### 4.1 - Linear Forward

![](https://i.imgur.com/LGn7wWI.png)

- linear_forward(A_prev, W, b)
    ```python
    def linear_forward(A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        ### START CODE HERE ### (≈ 1 line of code)
        Z = np.dot(W, A) + b
        ### END CODE HERE ###

        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache
    ```
    
    - 要注意cache裡面依序放：
    
        - ![](https://i.imgur.com/pLuVgq0.png)
        
        
### 4.2 - Linear-Activation Forward

![](https://i.imgur.com/Po3KuzJ.png)

- linear_activation_forward(A_prev, W, b, activation)

    - 顧名思義，將linear function和activation function包成一個function。
    - 所以input除了linear function需要的A_prev、W、b外，還需要額外提供選擇的啟動函數。
    
    ```python
    # GRADED FUNCTION: linear_activation_forward

    def linear_activation_forward(A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
            ### END CODE HERE ###

        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
            ### END CODE HERE ###

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache
    ```
    
    - output中的cache是由(linear_cache, activation_cache)組成的，由於這份作業沒有實作relu和sigmoid，個人不是很肯定activation_cache包什麼，不過直覺來應該是Z，因此：
    
        - ![](https://i.imgur.com/Pcaws91.png)
    
    
### 4.3 - L-Layer Model

![](https://i.imgur.com/vHDuD4Q.png)

- 本例中，除了最後一層啟動函數採用sigmoid之外，其餘都是relu。因此可以用一個for迴圈，重複計算前L-1層，再額外算最後一層，若未來有機會碰到每一層可能是不同的啟動函數時，合理的假設cache中也要包含啟動函數的資訊。

- L_model_forward(X, parameters)

    ```python
    # GRADED FUNCTION: L_model_forward

    def L_model_forward(X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            ### START CODE HERE ### (≈ 2 lines of code)
            A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], "relu")
            caches.append(cache)
            ### END CODE HERE ###

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (≈ 2 lines of code)
        AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b'+ str(L)], "sigmoid")
        caches.append(cache)
        ### END CODE HERE ###

        assert(AL.shape == (1, X.shape[1]))

        return AL, caches
    ```
    
    - AL是第L層的output，也是最終y的估計值
    - caches包含了每一層的cache: caches = [cache1, cache2,..., cacheL]
        - 每個cache包了：(liner_cache, activation_cache)
            - liner_cache: (A_prev, W, b)
            - activation: (Z)
    - 另外需要小心的是第0層沒有參數，因此caches[0]其實是第一層要用的cache。


## 5 - Cost function

![](https://i.imgur.com/nhNdMKG.png)

- compute_cost(AL, Y):

    ```python
    # GRADED FUNCTION: compute_cost

    def compute_cost(AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]

        # Compute loss from aL and y.
        ### START CODE HERE ### (≈ 1 lines of code)
        cost = - np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
        ### END CODE HERE ###

        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())

        return cost
    ```
    
    
## 6 - Backward propagation module

- 有Cost function，便能求導：dAL

    - 有dAL及啟動函數後，能求導：
        - dZL

    - 有dZL以及A^{[L-1]}，能求導：
        - dWL
        - dbL

    - 同時有dZL和W及b，能求導：
        - dA{L-1}

- 重複上面3個步驟，便能將全的參數都調較過一次。


### 6.1 - Activative backward

- relu_backward(dA, activation_cache):

    ```python
    # 有空再回來實做
    """
    Returns：
    dZ --- Gradient of the cost with respect to the linear output, when A = relu(Z).
    """
    ```

- sigmoid_backward(dA, activation_cache):

    ```python
    # 有空再回來實做
    """
    Returns：
    dZ --- Gradient of the cost with respect to the linear output, when A = sigmoid(Z).
    """
    ```


### 6.2 - Linear backward

![](https://i.imgur.com/TfucbZm.png)

- linear_backward(dZ, cache):

    ```python
    def linear_backward(dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        ### START CODE HERE ### (≈ 3 lines of code)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis = 1, keepdims = True)/m
        dA_prev = np.dot(W.T, dZ)
        ### END CODE HERE ###

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db
    ```
    
### 6.3 - Linear-Activation backward

![](https://i.imgur.com/ZueRMnO.png)

- linear_activation_backward(dA, cache, activation):

    ```python
    def linear_activation_backward(dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache

        if activation == "relu":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache)
            ### END CODE HERE ###

        elif activation == "sigmoid":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache)
            ### END CODE HERE ###

        return dA_prev, dW, db
    ```
    
    - 將activation_backward和linear_backward包起來變成，輸入dA以及cache，且dZ在內部已經給linear_backward使用，因此最終輸出：
        - dW
        - db
        - dA_prev


### 6.4 - L-Model Backward

![](https://i.imgur.com/6Q43pIQ.png)

- L_model_backward(AL, Y, caches):

    ```python
    def L_model_backward(AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        ### END CODE HERE ###

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[L - 1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
        ### END CODE HERE ###

        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            ### START CODE HERE ### (approx. 5 lines)
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###

        return grads
    ```
    
### 6.5 - Update Parameters

![](https://i.imgur.com/2Yy2ijb.png)

- update_parameters(parameters, grads, learning_rate):

    ```python
    def update_parameters(parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        ### START CODE HERE ### (≈ 3 lines of code)
        for l in range(L):
            parameters["W" + str(l+1)] -= grads["dW" + str(l+1)] * learning_rate
            parameters["b" + str(l+1)] -= grads["db" + str(l+1)] * learning_rate
        ### END CODE HERE ###
        return parameters
    ```
