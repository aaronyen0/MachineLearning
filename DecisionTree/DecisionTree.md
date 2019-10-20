# Decision Tree

---
## 參考資料
- [[資料分析&機器學習] 第3.5講 : 決策樹(Decision Tree)以及隨機森林(Random Forest)介紹](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda)

- [Wiki: Decision Tree](https://en.wikipedia.org/wiki/Decision_tree_learning)

## Introduction

![](https://i.imgur.com/OSIEhsY.png)

- 明確地表達決策過程
- 決策樹也稱作分類樹或回歸樹

## Information Gain
- 從樹根開始將資料的特徵將資料分割到不同邊，分割的原則是：分割要能得到最大的資訊增益(Information gain, 簡稱IG)。
![](https://i.imgur.com/Zq8K99I.png)

### 常見的資訊量有兩種：熵(Entropy) 以及 Gini不純度(Gini Impurity)
- ![](https://i.imgur.com/991PyST.png)

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
