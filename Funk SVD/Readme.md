# Funk SVD
傳統的SVD在實際運用中遇到了稀疏性問題和效率問題，他要求原始的矩陣必須要是稠密的，但現實中蒐集的資料矩陣往往都是稀疏的，其中存在許多的「洞」，讓許多傳統矩陣分解(decomposition, factorization)方法無法使用。


Funk SVD是一個經典的稀疏矩陣分解的方法，他最早公開在Simon Funk的部落格中，這方法至今仍是許多推薦系統的雛型。


## [Techniques]

### [Latent factor model模型]
假設今天有一個`矩陣R(m * n)，其中m為User的總數，n為Item的個數`，定義R(i, j)：User(i)對Item(j)的評分。可以合理的猜測，當User和Item數量夠多時，矩陣R會是一個稀疏矩陣。

Funk SVD方法中，假定User和Item可以用一些潛在類別作為分類依據，下圖假設潛在類別為3類：


![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ4.gif)


注意到，`矩陣R中的「？」代表了該格沒有任何資料，現在假定矩陣R可以分解為P(m * 3) * Q(3 * n)`。

此時P(i, :)可視為User(i)對各類別的偏好；Q(:, j)可視為Item(j)受到各類別的影響程度，當給定一組P和Q後，便能得到R的估計值：


![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ5.gif)


上式估計出來的R便是Funk SVD方法下，User(i)對Item(j)的估計評價。


### [目標函數]
有了上節的模型後，剩下的便是想辦法調整P和Q的係數，使得模型出來的估計值可以逼近真值，機器學習中調整參數最常見的方法，便是定義目標函數，或者稱為Loss funcion，而Funk SVD用來估計P和Q所使用的Loss funcion就是最小平方法，如下圖：


![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ3.gif)


有了Loss funcion，接著便能透過一些數值方法，如梯度下降法，尋找Loss function的極值，最終求得P和Q的估計值

    # 個人對本模型參數維度的一些想法
    
        Funk SVD總共有|M|個待估計的點(R中非空元素個數)，
        而參數個數為：n * classNum + m * classNum (|P| + |Q|)，
        最小平方法也可以說就是用這些參數構成|M|個Rhat，以norm2距離去逼近相對應的R，
        若|M|的數量過少，直覺來看可以得到多組甚至無數組(參數間線性相依)可以使SSE為0的解。
        例如：User和Item的個數都是1，假設classNum = 3，則|P|+|Q| = 3 + 3 = 6，
        用6個參數去估計一個R，會有無數種P,Q組合，都可以使 Rhat == R，
        但這些組合是否還有參考價值就相當耐人尋味。
        
        我目前正在觀看Coursera的機器學習課程(華盛頓大學)，
        課程中先後提到linear regression及logistic regression，
        兩者的上課範例都遇到參數比目標式更多的狀況，
        但是課程影片卻沒有提出來討論，
        連課後題目都寫參數越多模型效果越好，
        事實上在使用graphlab建模時，
        兩者都跳出參數過多模型不准的警告，
        如果計算linear regression並不是用梯度法而是使用反逆矩陣投影的話，
        就會因為linear dependent導致沒有反逆矩陣而出現error，
        沒提出這些是個人覺得這門課較嚴重的缺點之一。


### [模型調整]
#### [正則化(regularization)]
在維基百科的[Matrix factorization (recommender systems)](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))或是很多文章中都有提到，增加潛在因子的數量將改善個性化及預測品質，但當參數過多時，overfitting的問題就隨之而來，一種常見的解法是在目標函式中加入regularization terms，如下圖：

![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ6.gif)

本式包住矩陣P及Q的可以是[Frobenius norm或其他norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm)，這邊假設是Frobenius norm，我們可以注意到，當P或Q的某些元素過大或是過小時，會造成SSE急遽的增長，與我們的最小化SSE的目標相牴觸，進而抑制不讓元素異常過大或過小(也就是減緩fitting函式造成的異常數值)。

關於正則化的直觀理解，可以詳細閱讀[The Problem of Overfitting, Cost Function, ...](https://medium.com/@ken90242/machine-learning%E5%AD%B8%E7%BF%92%E6%97%A5%E8%A8%98-coursera%E7%AF%87-week-3-4-the-c05b8ba3b36f)，該文章用了實例，讓我們直覺的認知到正則化的影響。


        # 那篇文章中有一段我也覺得值得特別深思的問題，該文作者提到：
            「我在學習正規化的時候，
              對於為什麼只是加上個正規多項式就可以改善overfitting感到非常疑惑，
              畢竟我們根本不知道要降低哪一個θ值(feature)的影響力啊」
            
            該文對此也有一番獨到看法，但我覺得那個解釋很奇怪...
            
            關於這個問題我也提出一些見解(以下配合該文使用θ當作可調整的模型參數)：
            
                不論原始模型或調整後的模型，學習的過程就是在最小化目標函式(SSE)，
                在原始模型中為了去match每個點，某些θ可能非常大或是非常小，
                造成整個模型展開後形狀過度扭曲(overfitting)，
                這是我們今天不樂見的結果，
                因此我們才在模型中加入正則項去抑制θ。
                
                假設這邊的正則項是採用norm2距離來當作損失，
                其實我並不覺得這個正則項在一視同仁的打壓所有的θ，
                由於norm2的數學性質，
                加入正則項後，對太大或太小的θ，造成的影響比接近0的θ高出許多，
                最終θ會在原始模型損失及正則項帶來的抑制間取得一個平衡。
            
            
            另外我們都知道正則項的係數大小代表其在模型中的影響力，
            這時就有另一個值得深思的問題，
            到底怎麼才能找到恰當的係數呢？


#### [偏置]
回顧最開始的模型：

![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ5.gif)

所有的估計值都是User(i)和Item(j)之間的交互關係，實際上有很多性質可能是某些Item或是某些User獨有的，例如某些User給的評分整體就是比較低，或是某些Item製作精美，所有人的給予很高的評價，這些都可能是該User或是該Item獨有的性質，因此也有人將估計函式進一步擴增為：

![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ7.gif)

其中μ在該式表示整體偏差、大寫U代表某個User的偏差、大寫I代表某個Item的偏差。

如果有一些統計學的知識的話，會發現這個模型幾乎和二因子ANOVA(two-way ANOVA)一模一樣：


![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ8.gif)


唯一的差別就是交叉項的部分換成了潛在類別的內積。


#### [隱式反饋 & SVD++]
前述的所有模型，都需要具體的評分才能反應在模型參數上，
[待補]
