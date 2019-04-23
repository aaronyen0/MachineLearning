# Gradient descent for neural networks

## **Review**

- **Forward**

    ![](https://i.imgur.com/aGfc66G.png)


- **Parameters:**

    ![](https://i.imgur.com/LRlc5QE.png)
    
    
- **Cost function**

    ![](https://i.imgur.com/uh03CrP.png)


## **Gradient descent**
    
- **Forward propagation:**

    ![](https://i.imgur.com/YZDgWFg.png)


- **Backward propagation:**

    **Note : Assuming that g2 = σ**
    
    - **Layer2**

        ![](https://i.imgur.com/cLdGi47.png)

    - **Layer1**
    
        ![](https://i.imgur.com/loFZAIL.png)

    
## **Random Initialization**

![](https://i.imgur.com/UWH0dfP.png)


- ![](https://i.imgur.com/UkECsfJ.png)

    -  0.01是一個夠小的值，令W初始化為一個隨機又接近0的數值

    - 初始值要夠接近0的目的在於配合sigmoid或是tanh等啟動函數，在靠近0的地方有較大的斜率，避免收斂效率會太差

    - 對一個淺層(例如只有一層)的神經網路，選擇0.01是可行的一個解法，但是對一個夠深的網路來說，或許就有其他考量需要換另一種方式決定這個係數。

- ![](https://i.imgur.com/Ca7P5ri.png)


