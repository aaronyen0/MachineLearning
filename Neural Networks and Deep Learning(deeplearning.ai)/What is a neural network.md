# What is a neural network?

## Housing Price Prediction

假設下圖紅點表示房屋面積跟價格的關係，假設我們今天用一條直線去fit那些點，因此可以得到橘線，我又合理的判斷價格不可能低於0，因此假設當橘線碰到x軸時，的價格通通變為0，我們將這條橘線可以用下列的function表達。

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L1_1.GIF)

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L1_2.GIF)

我們可以換一個方式想像，假設這個函式就是一個最簡單的神經網路，如下圖：

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L1_3.GIF)

在類神經網路中，稱中間的圓為神經元(neuron)`(註：f(x) = max(0, reg(x))，我們常稱之為ReLU function (Rectified Linear Unit))`，想像我們今天用以預測價格的參數變得更多，除了area之外，還有bedrooms、zipCode、wealth。

其中area及bedrooms跟facility有關、zipCdoe跟walkability有關、zipCdoe跟wealth跟school quality有關，在神經網路的架構下，一種可能預測price的架構如下圖：

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L1_4.GIF)

<br><br>
一般而言，實務常見的神經網路架構如下圖，其中area、bedrooms、zipCode、wealth一樣是我們的輸入特徵；price則是我們感興趣的outcome。

當中的那些圓圈，我們又稱之為neural network的hidden units；而最左側的input層(area,bedrooms,...)稱為input layer；而f1~f3那層則稱之為density connected，代表所有輸入特徵都連接上中間層的所有圓圈。

值得注意的是只要有足夠的資料量，這種架構往往能自動train出很好的參數去fit那些資料

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L1_5.GIF)

<br><br>

## Supervised Learning with Neural Networks

### Neural Network examples

Input(x)|Output(y)|Application|Neural Network Type
----|----|----|----
Home features|Price|Real Estate|standard NN
Ad, user info|click on ad?|Online Advertising|standard NN
Image|Object(1,...,1000)|Photo tagging|CNN(convolutional NN)
Audio|Text transcript|Speech recognition|RNN(sequence data)
English|Chinese|Machine translation|RNN(sequence data)
Image, Radar info|Position of other cars|Autunomous driving|hybrid neural network(complex case)


![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L1_6.GIF)

<br>

### Structured Data / Unstructured Data
<br>

**Structured Data**
基本上就是資料庫的資料，也就是每個一特徵都是定義的好資料，如：

Area|#bedrooms|...|Price
:----:|:----:|:----:|:----:
2000|3|...|400
2500|2|...|350
⋮|⋮|⋮|⋮
1550|4|...|280


User Age|Ad ID|...|Click
:----:|:----:|:----:|:----:
41|93242|...|1
80|15516|...|0
⋮|⋮|⋮|⋮
22|00245|...|1

<br><br>

**Unstructured Data**

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L1_7.GIF)


## Why is Deep Learning taking off

隨著資料量到達某種程度後，很多傳統機器學習方法的performance便漸趨收斂；深度學習在這方面的表現往往比較好，可以接受更大量的資料並持續提升整體performance。下面示意圖中 紅線/藍線 分別代表 深度學習/傳統機器學習 資料數量對performance收斂情形。當資料量少時，各方法往往很難區分優劣的；但當資料量達一定程度時，比較一致的會發現到越大型的神經網路，會有更好的表現，但使用越大型的神經網路，一方面要有足夠資料，另一方面也需要有足夠的運算能力。

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L1_8.GIF)

而過去十幾年來，很多訓練大型神經網路的必要條件漸趨成熟：
- Data
- Computation
- Algorithms
    - 有很多創新演算法讓訓練速度更快
    - 如用 ReLU function 取代 sigmoid function，解決了sigmoid function在離0比較遠的地方，梯度太小造成參數收斂太慢的問題。<br> 
    ![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L1_9.GIF)
    - 設計良好的演算法，可以推進實驗循環的速度。<br>
    ![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L1_10.GIF)



