# Funk SVD
傳統的SVD在實際運用中遇到了稀疏性問題和效率問題，他要求原始的矩陣必須要是稠密的，但現實中蒐集的資料矩陣往往都是稀疏的，其中存在許多的「洞」，讓許多傳統矩陣分解(decomposition, factorization)方法無法使用。<br>

Funk SVD是一個經典的稀疏矩陣分解的方法，他最早公開在Simon Funk的部落格中，這方法至今仍是許多推薦系統的雛型。<br>

## [Techniques]

### [Latent factor model模型]
假設今天有一個`矩陣R(m * n)，其中m為User的總數，n為Item的個數`，定義R(i, j)：User(i)對Item(j)的評分。可以合理的猜測，當User和Item數量夠多時，矩陣R會是一個稀疏矩陣。<br><br>
Funk SVD方法中，假定User和Item可以用一些潛在類別作為分類依據，下圖假設潛在類別為3類：<br>
![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ4.gif)<br><br>
注意到，`矩陣R中的「？」代表了該格沒有任何資料，現在假定矩陣R可以分解為P(m * 3) * Q(3 * n)`。<br>
此時P(i, :)可視為User(i)對各類別的偏好；Q(:, j)可視為Item(j)受到各類別的影響程度，當給定一組P和Q後，便能得到R的估計值：<br>
![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ5.gif)<br>
上式估計出來的R便是Funk SVD方法下，User(i)對Item(j)的估計評價。<br>

### [目標函數]
有了上節的模型後，剩下的便是想辦法調整P和Q的係數，使得模型出來的估計值可以逼近真值，<br>
機器學習中調整參數最常見的方法，便是定義目標函數，或者稱為Loss funcion<br>
而Funk SVD用來估計P和Q所使用的Loss funcion就是最小平方法，如下圖：<br>
![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ3.gif)<br>
有了Loss funcion，接著便能透過一些數值方法，如梯度下降法，尋找Loss function的極值，最終求得P和Q的估計值<br>

    # 個人對於本模型中維度的看法
        從維度的角度來看，總共有|M|個待估計的點(R中非空元素個數)，
        而參數個數為：n * classNum + m * classNum (|P| + |Q|)，
        最小平方法其實就是用這些參數組成了|M|個Rhat，以norm2距離去逼近相對應的R，
        若|M|的數量過少，直覺來看應該可以得到一組甚至無數組(參數間線性相依)可以使SSE為0的解，
        例如：User和Item的個數都是1，假設classNum = 3，則|P|+|Q| = 3 + 3 = 6，此時會有無數種P,Q組合，使 Rhat == R。
        這些組合是否還有參考價值就有點耐人尋味。
<br><br>
### [避免過度擬合的調整模型]
  

