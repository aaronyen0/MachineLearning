# Shallow Neural Network - Activation functions

## Introduction

- **Recall a Shallow Neural Network**

    ![](https://i.imgur.com/RD0yC98.gif)
    
    在前面的章節中，給定一個網路架構及input x後，可以推出下圖的流程：
    
    ![](https://i.imgur.com/GZhc6Ds.png)

    其中σ在這裡扮演activation function的角色，在此之前都是使用sigmoid function，事實上根據不同需求，我們可以在不同的層使用不同的activation function，我們將流程改寫成：
    
    ![](https://i.imgur.com/oa7THvq.png)
    
    其中g^[1] 或是g^[2] 分別代表模型中Layer1和Layer2所採用的activative function。


## Activative function

- **機器學習中在挑選啟動函數時，通常希望滿足某些性質：**

    - **Nonlinear**
    When the activation function is non-linear, then a two-layer neural network can be proven to be a universal function approximator. The identity activation function does not satisfy this property. When multiple layers use the identity activation function, the entire network is equivalent to a single-layer model.

    - **Range**
    When the range of the activation function is finite, gradient-based training methods tend to be more stable, because pattern presentations significantly affect only limited weights. When the range is infinite, training is generally more efficient because pattern presentations significantly affect most of the weights. In the latter case, smaller learning rates are typically necessary.

    - **Continuously differentiable**
    This property is desirable (RELU is not continuously differentiable and has some issues with gradient-based optimization, but it is still possible) for enabling gradient-based optimization methods. The binary step activation function is not differentiable at 0, and it differentiates to 0 for all other values, so gradient-based methods can make no progress with it.

    - **Monotonic**
    When the activation function is monotonic, the error surface associated with a single-layer model is guaranteed to be convex.

    - **Smooth functions with a monotonic derivative**
    These have been shown to generalize better in some cases.

    - **Approximates identity near the origin**
    When activation functions have this property, the neural network will learn efficiently when its weights are initialized with small random values.When the activation function does not approximate identity near the origin, special care must be used when initializing the weights.
        
    **參考資料：**
    - [Activation Functions](https://towardsdatascience.com/activation-functions-in-neural-networks-58115cda9c96)
    - [Activation function - wiki](https://en.wikipedia.org/wiki/Activation_function#cite_note-7)
    - [why activation functions that approximate the identity near origin are preferabl](https://stats.stackexchange.com/questions/288722/why-activation-functions-that-approximate-the-identity-near-origin-are-preferabl)

- **Identity or linear activation function**
    - Input maps to same output.
    
    ![](https://i.imgur.com/Obso3w3.png)

- **Binary Step**
    - 常見於分類器中
    
    ![](https://i.imgur.com/vgrie4E.png)

- **Logistic or Sigmoid**
    - in rnage (0,1)
    - 常見於神經網路
    
    ![](https://i.imgur.com/Tv3LFBy.png)

- **Tanh**
    - in range (-1,1)
    - 比sigmoid更陡峭
    
    ![](https://i.imgur.com/JRnJvxa.png)

- **ArcTan**
    - in range (-pi/2, pi/2)
    - Similar to sigmoid and tanh function
    
    ![](https://i.imgur.com/ua5jiWp.png)

- **Rectified Linear Unit (ReLu)**
    - 除去落於負數的部分
    
    ![](https://i.imgur.com/5hSMMtA.png)

- **Leaky ReLu**
    - 相較於ReLu，落於負數的部分有正斜率，不過相較於正數的部分低
    - 用來解決ReLu中落於負數的因子從此失效的問題
    
    ![](https://i.imgur.com/EUgn9jg.png)

- **Softmax**
    - Softmax function is used to impart probabilities when you have more than one outputs you get probability distribution of outputs.
    - Useful for finding most probable occurrence of output with respect to other outputs.
    - 其實是sigmoid的變形，也可以說是多維的logistic function
    ![](https://i.imgur.com/LNSlrBr.png)

## [Choose a proper activation function](https://www.quora.com/How-should-I-choose-a-proper-activation-function-for-the-neural-network#Roygk)

該文提到，現實應用中，我們可能在不同的架構中，使用不同的啟動函數。其實很難知道什麼樣的函數才是真正適合這個架構，因此多閱讀相關文獻，才是掌握這些的不二法門。文中另外舉了幾個例子並簡單解釋何時使用他們：

- Linear：
    - good choice for regression problem

- ReLU/ELU
    - I want to see only what I am looking for
    - When the incoming signal hits the neuron, it can be recognized as “less relevant” or “more relevant”.
    - By zeroing “less relevant” signals, ReLU acts as an excitatory neuron that reacts on relevant signals and passes the information about them further. 

- Softmax
    - give me the probability distribution
    - If you know that whatever comes out of the layer must be a distribution, this is what you want to use.

- Tanh
    - In some cases the sign of the output is relevant, but the magnitude can mess with the further computations.
    -  It’s useful when after the magnitude of unprocessed output grows significantly, the further growth is not that important, and vice versa, when the fluctuations around zero make significant difference.
