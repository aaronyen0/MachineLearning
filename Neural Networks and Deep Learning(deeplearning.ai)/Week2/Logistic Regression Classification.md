# Logistic Regression and Gradient Descent

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

推薦：[**線代啟示錄**](https://ccjou.wordpress.com/)是周老師的部落格，我從線代啟示錄上獲得過許多的幫助及啟發。除了精美的圖例講解外，各主題都有非常詳細的證明，不論是否為數學愛好者，都可以在當中獲得需要的資訊。


## Gradient Descent

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic13.GIF)


[梯度下降法](https://en.wikipedia.org/wiki/Gradient_descent)：
如果實值函數F(x)在點 a 處`可微且有定義`。
那麼函數F(x)在 a 點，沿著`梯度`相反方向，也就是-▽F(a)方向的下降速率最快。
因此對alpha > 0且數字夠小時，假設 b = a - alpha * ▽F(a)，則 F(b) <= F(a)。


回到本例，給定一組初始值a，可得J在a點的梯度：

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic14.GIF)

### 學習效率 alpha

若還記得當初怎麼學習微積分，一定不會對以下的敘述陌生：當函數在某個點附近滿足某些條件，對所有足夠小的epsilon，以這個點為中心，半徑epsilon的範圍內，這個函數會維持著某些性質...bla bla bla

在梯度下降法裡面，alpha間接扮演epsilon的角色，我們知道往梯度方向`走一點點`可以找到更佳的解，但是卻不知這一點點容許範圍多少，alpha太小會造成收斂效率極差，alpha太大則會造成參數震盪甚至整個發散，之前看過某些backtrack的方法是先做迴測再決定alpha值(特別在有限制式的最佳化問題，為了怕變數超出限制式範圍，常會先測試alpha是否合理)，下面舉一些簡單的圖例：

- 下圖 z0初始值皆是0，也就是最左上角的點
- 左上：可以發現alpha很小的時候要疊代很多次才會收斂
- 右上：隨著alpha稍大之後收斂速度就提高了
- 左下：本例的alpha在超過0.5後，變數開始震盪，不過還是會收斂
- 右下：alpha接近1的時候，變數震盪的很明顯，不過依然會收斂

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic15.GIF)

- 下圖 alpha = 1.1、z0初始值是20，也就是最下面的點
- 由於alpha設得太大，造成步寬太遠，反而造成疊代後越來越發散

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic16.GIF)

- 在這些case中，alpha設0.6是收斂最快的(有震盪)，但設0.3可能是最理想的。
- 不過那也只是本例太過單純且容易觀察，事實上DP架構中動輒上萬參數，縱使是分層疊代，但在同一層中要找到合適的alpha也是相當有挑戰性的事情

#### [**牛頓法**](https://en.wikipedia.org/wiki/Newton%27s_method)

大家都知道在連續可微函數中，若存在極值，其梯度必然是0，因此上述的梯度下降法問題，可以轉換成梯度為0的問題，這個時候有一個很常見的數值方法叫牛頓法。

牛頓法其實也是梯度法的一種，他較特別的地方在，牛頓法是根據導數的值來決定步長，因此我們並不用擔心alpha值設多少，但是相較梯度下降法，個人覺得牛頓法對函數的特性和初始值的要求比較嚴格。 (例如：縱使是一個嚴格遞減或遞增的函數，若在疊代範圍及解的附近，二階導數沒有維持同方向，則很可能發散掉，這種發散方式有點類似經濟學中的[蛛網理論](https://wiki.mbalib.com/zh-tw/%E8%9B%9B%E7%BD%91%E7%90%86%E8%AE%BA))。

### 初始值

只要是梯度法，就會遇到初始值的問題，不難想像在不同的初始位置，最終會落在不同的洞裡。

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic17.GIF)

常與梯度法互相映照的就是一些隨機打點方法，隨機打點適合用在函數較複雜或是難以微分的時候，常見如基因演算法、PSO...等。

雖然隨機打點方法同樣有初始值的問題，但隨著隨機性提高，可以較廣域的搜尋極值，但就我所知，一般公司並不太願意使用這些方法，主要原因之一是，隨機打點在同一個情境下，有機會產生不同結果。



## Gradient Descent on n examples

![](https://github.com/worcdlo/MachineLearning/blob/master/Neural%20Networks%20and%20Deep%20Learning(deeplearning.ai)/Teaching%20Material/L2_pic18.GIF)


<br><br><br><br><br><br><br><br>
