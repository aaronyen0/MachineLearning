# Funk SVD
傳統的SVD在實際運用中遇到了稀疏性問題和效率問題，他要求原始的矩陣必須要是稠密的，但現實中蒐集的資料矩陣往往都是稀疏的，其中存在許多的「洞」，讓許多傳統矩陣分解(decomposition, factorization)方法無法使用。<br>

Funk SVD是一個經典的稀疏矩陣分解的方法，他最早公開在Simon Funk的部落格中，這方法至今仍是許多推薦系統的雛型。<br>

## [Techniques]

假設今天有一個`R(m*n)，其中m為User的總數，n為Item的個數`，定義R(i,j)：User(i)對Item(j)的評分。可以合理的猜測，當User和Item數量夠多時，矩陣R會是一個稀疏矩陣。<br><br>
Funk SVD方法中，假定User和Item可以用一些潛在類別作為分類依據，下圖假設潛在類別為3類：<br><br>
![](https://github.com/worcdlo/Machine-Learning/blob/master/Funk%20SVD/Equ4.gif)<br><br>
令`矩陣R可以分解為P(m*3) * Q(3*n)`，<br>
此時P(i,:)可視為User(i)對各類別的偏好；Q(:,j)可視為Item(j)受到各類別的影響程度，<br>
因此R(i,j) = <P(i,:), Q(:,j)>，便是User(i)對Item(j)的整體評價。<br>
