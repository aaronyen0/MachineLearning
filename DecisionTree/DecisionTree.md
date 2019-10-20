# Decision Tree

---
## 參考資料
- [[資料分析&機器學習] 第3.5講 : 決策樹(Decision Tree)以及隨機森林(Random Forest)介紹](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda)

- [Wiki: Decision Tree](https://en.wikipedia.org/wiki/Decision_tree_learning)

- [决策树之CART算法](https://blog.csdn.net/ACdreamers/article/details/44664481)

## Introduction

![](https://i.imgur.com/OSIEhsY.png)

- 明確地表達決策過程
- 決策樹也稱作分類樹或回歸樹

## Information Gain
- 從樹根開始將資料的特徵將資料分割到不同邊，分割的原則是：分割要能得到最大的資訊增益(Information gain, 簡稱IG)。
![](https://i.imgur.com/Zq8K99I.png)

### 常見的資訊量有兩種：熵(Entropy) 以及 Gini不純度(Gini Impurity)

- Entropy
    - ![](https://i.imgur.com/6EIIjmp.png)

- Gini
    - ![](https://i.imgur.com/MASpUpW.png)


- Example

    |有房(x1)|婚姻(x2)|年收(x3)|拖欠貸款(y)|
    |-|-|-|-|
    |yes|single|125K|no|
    |no|married|100K|no|
    |no|single|70K|no|
    |yes|married|120K|no|
    |no|divorced|95K|yes|
    |no|married|60K|no|
    |yes|divorced|220K|no|
    |no|single|85K|yes|
    |no|married|75K|no|
    |no|single|90K|yes|

    - Split by marriage
        - Case1: single vs. others

            | |single|others|
            |-|-|-|
            |no default|2|5|
            |default|2|1|
        
        - Case2: married vs. others

            | |married|others|
            |-|-|-|
            |no default|4|3|
            |default|0|3|
        
        - Case3: divorced vs. others

            | |divorced|others|
            |-|-|-|
            |no default|1|6|
            |default|1|2|
        
    - Entropy
        - I(D) = - [0.7\*log(0.7) + 0.3\*log(0.3)] = 0.88
        - Case1: single vs. others
            - I(single) = - [0.5\*log(0.5) + 0.5\*log(0.5)] = 1.0
            - I(others) = - [0.83\*log(0.83) + 0.17\*log(0.17)] = 0.65
            - IG = 0.88 - 0.4\*1.0 - 0.6\*0.65 = 0.09
            
        - Case2: married vs. otehrs
            - I(married) = - [1.0\*log(1.0) + 0.0\*log(0.0)] = 0.0
            - I(otehrs) = - [0.5\*log(0.5) + 0.5\*log(0.5)] = 1.0
            - IG = 0.88 - 0.4\*0.0 - 0.6\*1.0 = 0.28
        
        - Case3: divorced vs. others
            - I(divorced) = - [0.5\*log(0.5) + 0.5\*log(0.5)] = 1.0
            - I(others) = - [0.75\*log(0.75) + 0.25\*log(0.25)] = 0.81
            - IG = 0.88 - 0.2\*1.0 - 0.8\*0.81 = 0.03

    - Gini Impurity
        - I(D) = 1 - 0.7^2 - 0.3^2 = 0.42
        - Case1: single vs. others
            - I(single) = 1 - 0.5^2 - 0.5^2 = 0.5
            - I(others) = 1 - 0.83^2 - 0.17^2 = 0.28
            - IG = 0.42 - 0.4\*0.5 - 0.6\*0.28 = 0.05
        - Case2: married vs. otehrs
            - I(single) = 1 - 1.0^2 - 0.0^2 = 0.0
            - I(others) = 1 - 0.5^2 - 0.5^2 = 0.5
            - IG = 0.42 - 0.4\*0.0 - 0.6\*0.5 = 0.12
            
        - Case3: divorced vs. others
            - I(single) = 1 - 0.5^2 - 0.5^2 = 0.5
            - I(others) = 1 - 0.75^2 - 0.25^2 = 0.375
            - IG = 0.42 - 0.2\*0.5 - 0.8\*0.375 = 0.02

    - Comparity: **Information Gain** on marriage
        
        ||Entropy|Gini Impurity|
        |-|-|-|
        |single|0.09|0.05|
        |married|**0.28**|**0.12**|
        |divorced|0.03|0.02|
        
        - 兩種標準都認為用married切割節點，能讓IG最大

## CART (Classification And Regression Trees)
CART算法是一種二分遞歸分割技術，把當前樣本劃分為兩個子樣本，使得生成的每個非葉子結點都有兩個分支。

因此CART算法生成的決策樹是結構簡潔的二叉樹。由於CART算法構成的是一個二叉樹，它在每一步的決策時只能分為「是」或「否」，即使一個feature有多個取值，也是把數據分為兩部分。在CART算法中主要分為兩個步驟

- 將樣本遞歸劃分進行建樹過程
- 用驗證數據進行剪枝

### CART算法的原理
設(x_1, ..., x_m)代表單個樣本的m種屬性，y表示所屬類別。 CART算法通過遞歸的方式將m維的空間劃分為不重疊的矩形。劃分步驟大致如下

- (1) 選一個自變量x_i，再選取x_i中的一個值v，v把m維空間劃分為兩部分，一部分的所有點都滿足x_i <= v，另一部分的所有點都滿足x_i > v，對非連續變量來說屬性值的取值只有兩個，即等於該值或不等於該值。

- (2) 遞歸處理，將上面得到的兩部分按步驟(1)重新選取一個屬性繼續劃分，直到把整個維空間都劃分完。
