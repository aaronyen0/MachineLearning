# Logistic Regression

## Define

假設有n組歷史資料，每一組歷史資料有k個變數(independent variable)，並對應到一個Y(dependent variable)，且Y值只有0或1，分別為互斥的兩個事件之代號(ex:選或不選)：

![](https://github.com/worcdlo/MachineLearning/blob/master/Models%20For%20Discrete%20Choice/logit1.GIF)

Logistic Regression Model被定義為： `Y_i之間彼此獨立，且Y_i滿足Bernoulli(p_i)分配，同時p_i的log-odds可以被表達為一個由x_i構成的線性模型。`

![](https://github.com/worcdlo/MachineLearning/blob/master/Models%20For%20Discrete%20Choice/logit2.GIF)

其中截距項Beta_0代表log-odds表示下的baseline，而Beta_j則代表自變數j(註:第i組資料的第j個自變數命名為x_ij)對log-odds的正/負相關程度，
我們可以將上式轉換為另一個表示式：

![](https://github.com/worcdlo/MachineLearning/blob/master/Models%20For%20Discrete%20Choice/logit3.GIF)

下圖假定 k = 1，給定n組樣本下，估計出來的示意圖，其中紅點代表該樣本資料，而黑色曲線則是以這些紅點估計出來的p(x)。

![](https://github.com/worcdlo/MachineLearning/blob/master/Models%20For%20Discrete%20Choice/logit5.GIF)

## Statistical Inference

