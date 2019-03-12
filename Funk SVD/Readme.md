# Funk SVD
傳統的SVD在實際運用中遇到了稀疏性問題和效率問題，他要求原始的矩陣必須要是稠密的，但現實中蒐集的資料矩陣往往都是稀疏的，其中存在許多的「洞」，讓許多傳統矩陣分解(decomposition, factorization)方法無法使用。<br>

Funk SVD是一個經典的稀疏矩陣分解的方法，他最早公開在Simon Funk的部落格中，這方法至今仍是許多推薦系統的雛型。<br>

## [Techniques]

### [模型]
假設今天有一個`矩陣R(m * n)，其中m為User的總數，n為Item的個數`，定義R(i, j)：User(i)對Item(j)的評分。可以合理的猜測，當User和Item數量夠多時，矩陣R會是一個稀疏矩陣。<br><br>
Funk SVD方法中，假定User和Item可以用一些潛在類別作為分類依據，下圖假設潛在類別為3類：<br>
![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ4.gif)<br><br>
注意到，`矩陣R中的「？」代表了該格沒有任何資料，現在假定矩陣R可以分解為P(m * 3) * Q(3 * n)`。<br>
此時P(i, :)可視為User(i)對各類別的偏好；Q(:, j)可視為Item(j)受到各類別的影響程度，當給定一組P和Q後，便能得到R的估計值：<br>
![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ5.gif)<br><br>
便是Funk SVD方法下，User(i)對Item(j)的估計評價。<br><br>

### [目標函數]
有了上節的模型後，剩下的便是想辦法調整P和Q的係數，使得模型出來的估計值可以逼近真值，<br>
機器學習中調整參數最常見的方法，便是定義目標函數，或者稱為Loss funcion：<br>
![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ3.gif)<br><br>
有了目標函數，接著便能透過一些數值方法，如梯度下降法，求得最佳的P和Q的值<br><br>
