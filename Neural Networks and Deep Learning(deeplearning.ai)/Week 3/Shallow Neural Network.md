# Shallow Neural Network

## Recall Logistic Regression

**sigmoid function 在本例中就是所謂的 啟動函數(activation function)**

![](https://i.imgur.com/McZS39d.gif)

- **Output Layer**

    $z = \mathbf{w^T x} + b, \ \hat{y} =   a = \frac{1}{1 + e^{-z}}$

- **Loss Funciton**

    $L(a, y)$

## Shallow Nueral Network

**註：[j] 上標中括號內的數字j，代表是神經網路中的第j層**

![](https://i.imgur.com/PwAMt7C.gif)

上圖是一個兩層的神經網路，其中：

- **Intput Layer**
    
    $x_1, \  x_2, \ x_3, \ ... \ , \ x_k$

- **[1]第一層(Hidden Layer)**：

    $z^{[1]}_1 = \mathbf{{w^{[1]}_1}^T x}+b^{[1]}_1$, $a^{[1]}_1 = \frac{1}{1 + e^{-z^{[1]}_1}}$

    $z^{[1]}_2 = \mathbf{{w^{[1]}_2}^T x}+b^{[1]}_2$, $a^{[1]}_2 = \frac{1}{1 + e^{-z^{[1]}_2}}$

    ...

    $z^{[1]}_q = \mathbf{{w^{[1]}_q}^T x}+b^{[1]}_q$, $a^{[1]}_q = \frac{1}{1 + e^{-z^{[1]}_q}}$
    
    
    展開成矩陣形式，令 $\mathbf{z^{[1]}} = \mathbf{W^{[1]}x} + \mathbf{b^{[1]}}$：
    
    $\begin{bmatrix} z^{[1]}_1\\z^{[1]}_2\\ ⋮\\z^{[1]}_q\end{bmatrix}= \begin{bmatrix} - & \mathbf{{w^{[1]}_1}^T} & -\\ - &\mathbf{{w^{[1]}_2}^T}&-\\ & ⋮ & \\ - & \mathbf{{w^{[1]}_q}^T} & -\end{bmatrix} \begin{bmatrix}x_1\\ x_2\\⋮ \\x_k\end{bmatrix} + \begin{bmatrix}b^{[1]}_1\\ b^{[1]}_2\\⋮ \\b^{[1]}_q\end{bmatrix}$
    
    $\mathbf{a^{[1]}} = \begin{bmatrix}a^{[1]}_1\\ a^{[1]}_2\\⋮ \\a^{[1]}_q\end{bmatrix} = \begin{bmatrix}σ(z^{[1]}_1)\\σ(z^{[1]}_2)\\⋮ \\σ(z^{[1]}_q)\end{bmatrix}$

    
    其中 $\mathbf{W}_{q*k}$ 可視為是一個特徵轉換矩陣(線性轉換矩陣)，新特徵(q維)的每個元素都受到所有input(k維)影響(因此有q*k個參數)，整個轉換其實就是將k維特徵轉換成q維特徵。

- **[2]第二層(Output Layer)：**

    $z^{[2]} = \mathbf{{w^{[2]}}^T} \begin{bmatrix} a^{[1]}_1\\a^{[1]}_2\\ ⋮\\a^{[1]}_q\end{bmatrix} + b^{[2]}$
    
    $a^{[2]} = \frac{1}{1 + e^{-z^{[2]}}} = σ(z^{[2]})$

- **Loss Funciton**

    $L(a^{[2]}, y)$
    
    

## Vectorizing across multiple examples

**註：(i) 上標小括號的數字i，代表第i組樣本。因此$a^{[j] (i)}$代表，第i組樣本在第j層的特徵向量。**

- **Pseudo Code**

    **for i = 1 to n:**
    &emsp;&emsp;$\mathbf{z^{[1](i)} = W^{[1]} \ x^{(i)} + b^{[1]}}$
    &emsp;&emsp;$\mathbf{a^{[1](i)} = σ(z^{[1](i)})}$
    &emsp;&emsp;$\mathbf{z^{[2](i)} = W^{[2]} \ a^{[1](i)} + b^{[2]}}$
    &emsp;&emsp;$\mathbf{a^{[2](i)} = σ(z^{[2](i)})}$

- **變數解釋：**

    - Input對每組樣本的展開式：

        $\mathbf{X_{k*n}} =\begin{bmatrix}
        | & | & | & | 
        \\
        x^{(1)} & x^{(2)} & ... & x^{(n)}
        \\
        | & | & | & | 
        \end{bmatrix}_{k*n}$

    - 第一層將Input線性轉換後的展開式：

        $\mathbf{Z^{[1]}_{q*n}} =\begin{bmatrix}
        | & | & | & | 
        \\
        z^{[1](1)} & z^{[1](2)} & ... & z^{[1](n)}
        \\
        | & | & | & | 
        \end{bmatrix}_{q*n}$

    - 第一層線性轉換中的截距向展開式(要小心b是要調整的參數，因此每行數字都一樣，代表對所有樣本都執行同樣的轉換運算)：

        $\mathbf{B^{[1]}_{q*n}} =\begin{bmatrix}
        | & | & | & | 
        \\
        b^{[1]} & b^{[1]} & ... & b^{[1]}
        \\
        | & | & | & | 
        \end{bmatrix}_{q*n}$

    - 第一層將Z經過啟動函數之後，出來的結果，這個結果將會是下一層的Input：

        $\mathbf{A^{[1]}_{q*n}} =\begin{bmatrix}
        | & | & | & | 
        \\
        a^{[1](1)} & a^{[1](2)} & ... & a^{[1](n)}
        \\
        | & | & | & | 
        \end{bmatrix}_{q*n}$

    - **第一層最終得到：**

        $\left\{\begin{matrix}
        \mathbf{Z^{[1]}_{q*n} = W^{[1]}_{q*k} X_{k*n}+ B^{[1]}_{q*n}}
        \\
        \\
        \mathbf{A^{[1]}_{q*n}} = σ(\mathbf{Z^{[1]}_{q*n}})
        \end{matrix}\right.$

    - **同理，第二層會得到：**

        $\left\{\begin{matrix}
        \mathbf{Z^{[2]} = W^{[2]} A^{[1]}+ B^{[2]}}
        \\
        \\
        \mathbf{A^{[2]}} = σ(\mathbf{Z^{[2]}})
        \end{matrix}\right.$



    
    
