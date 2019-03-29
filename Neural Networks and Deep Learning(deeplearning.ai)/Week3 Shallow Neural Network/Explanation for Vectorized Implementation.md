# Shallow Neural Network - Explanation for Vectorized Implementation

## Recall Logistic Regression

**sigmoid function 在本例中就是所謂的 啟動函數(activation function)**

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L3_pic01.GIF)

## Shallow Nueral Network

**註：[j] 上標中括號內的數字j，代表是神經網路中的第j層**

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L3_pic02.GIF)

上圖是一個兩層的神經網路，其中：

- **Intput Layer**
    
    ![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L3_pic03.GIF)

- **[1]第一層(Hidden Layer)**：

    ![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L3_pic04.GIF)
    
    其中 W_{q*k} 可視為是一個特徵轉換矩陣(線性轉換矩陣)，新特徵(q維)的每個元素都受到所有input(k維)影響(因此有q*k個參數)，整個轉換其實就是將k維特徵轉換成q維特徵。

- **[2]第二層(Output Layer)：**

    ![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L3_pic05.GIF)

- **Loss Funciton**

    ![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L3_pic06.GIF)
    
    

## Vectorizing across multiple examples

**註：(i) 上標小括號的數字i，代表第i組樣本。因此$a^{[j] (i)}$代表，第i組樣本在第j層的特徵向量。**

- **Pseudo Code**

    ![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L3_pic07.GIF)

- **變數解釋：**

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L3_pic08.GIF)

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L3_pic09.GIF)




    
