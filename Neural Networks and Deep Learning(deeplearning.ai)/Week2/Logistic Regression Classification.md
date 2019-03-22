# Logistic Regression as a Neural Network 等整篇上完再決定正式主題

# Classification

注意：個人習慣用n代表樣本數、k或m代表feature數，和課程影片剛好相反，若日後有人一邊看影片一邊對照筆記，請小心不要混淆。

## Notation

Given n pairwise datas:

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic1.GIF)

將所有x展開為一個大矩陣X，其中row代表features，column代表不同樣本：

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic2.GIF)

將所有y併為一個1\*n的向量：

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic3.GIF)

文中提到有些文獻中會用X_{n\*k}作為展開，也就是row代表不同樣本，column代表features，但影片中特別強調在建置神經網路時，X_{k\*n}的表達法會比X_{n\*k}更為恰當。

註：一般而言，小寫粗體字為向量；大寫粗體字為矩陣。

## Example

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic4.GIF)

### Linear Regression

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic5.GIF)

顯然這並不是一個適當的假設，我們都知道機率必然要落在0~1之間，hat{y}又代表y = 1時的機率，而線性模型並不會保證這件事情成立。

### Logistic Regression & Sigmoid Function

一種此情境下較常使用的Logistic Regression，使用的函式稱為sigmoid function：

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic6.GIF)

可以發現到當z很大時，sigmoid(z)逼近於1；z很小時，sigmoid(z)逼近於0。


### Redefine Expression of Parameters

為了方便用內積表示模型，我們將x向量和w向量重新定義如下，同樣的表達法在多變數迴歸中也常出現：

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic7.GIF)

## Loss Function & Coss Function

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic8.GIF)

### Loss Function

Loss function用來衡量單一樣本的訓練效果好不好，最常見的Loss function如下(平方差)：

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic9.GIF)

不過這個case並不適合用square error計算誤差，實際上在logistic regression中的loss function會定義為：

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic10.GIF)

下列敘述為何這個式子是make sense的：

若 y = 1，第二項的(1-y)為0，同時我們希望第一項中的yHat越接近1越好
若 y = 0，第一項的 y 為0，同時我們希望第二項中的(1-yHat)越接近1越好，等價於yHat越靠近0越好

不過Logistic Regression中較正統的Loss Function推導，其實是用Bernoulli的pdf計算MLE得到的。請參閱([**Logistic Regression**](https://github.com/worcdlo/MachineLearning/blob/master/Models%20For%20Discrete%20Choice/Logistic%20Regression.md))


### Cost Function

Cost function在衡量整體模型的訓練效果，這邊採用：

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic11.GIF)


### Compare different loss function

本來想要自己打點畫圖的，不過已經有前人先畫過，因此就直接借來用：
[Logistic regression cost surface not convex](https://stats.stackexchange.com/questions/267400/logistic-regression-cost-surface-not-convex)

![](https://i.imgur.com/QlHsBGO.gif)

圖中的Sum-of-Squared Loss和Log Loss分別是上述提過的兩種不同loss function所構成的cost function，而[Hinge Log](https://blog.csdn.net/hustqb/article/details/78347713)，則是另一種loss function，通常被用在最大間隔(maximum-margin)的二分類算法，如[SVM](https://en.wikipedia.org/wiki/Support-vector_machine)。

另外Hinge Loss是convex function，雖然他有些離散點不可微，但是依然可以用梯度下降法取得整個cost function的極值，關於凸函數和極值的關係，請詳閱參考資料。

參考資料：
- [**Hinge loss - Wikipedia**](https://en.wikipedia.org/wiki/Hinge_loss)
- [**凸函數 - 線代啟示錄**](https://ccjou.wordpress.com/2013/08/27/%E5%87%B8%E5%87%BD%E6%95%B8/)
- [**最佳化理論與正定矩陣 - 線代啟示錄**](https://ccjou.wordpress.com/2009/10/06/%E6%9C%80%E4%BD%B3%E5%8C%96%E5%95%8F%E9%A1%8C%E8%88%87%E6%AD%A3%E5%AE%9A%E7%9F%A9%E9%99%A3/)

推薦：[**線代啟示錄**](https://ccjou.wordpress.com/)是周老師的部落格，我從線代啟示錄上獲得到過許多的幫助及啟發。除了精美的圖例講解外，各主題都有非常詳細的證明，不論是否為數學愛好者，都可以在當中獲得需要的資訊。


## Gradient Descent

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic12.GIF)



<br><br><br><br><br><br><br><br>
