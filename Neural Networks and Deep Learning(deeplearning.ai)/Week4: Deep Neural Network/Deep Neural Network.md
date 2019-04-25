# Deep Neural Network

## dimension

## Why deep representations?

- 影像處理中，從關注在很微小的區域漸漸地匯集成大的區域

    - 第一層常常像是在找圖形的邊緣，線條方向等

    - 進入第二層，將第一層的數據匯集在一起，成為部分的特徵，例如眼睛、鼻子等

    - 最後將各種特徵又匯集在一起，形成完整的輪廓，例如人臉等

    ![](https://i.imgur.com/3PMuVLl.png)

- 語音辨識系統 Audio

    - 偵測低階語音波型，像是音調上升下降、音階等

    - 結合成低階音波特徵，學習偵測音的的基本單元，如母音、音位等

    - 組合在一起就變成了單字

    - 再組合在一起就變成了句子


## Circuit theory and deep learning

Informally: There are functions you can compute with a "small" L-layer deep neural network that shallower networks require exponentially more hidden units to compute;

- 課程舉的是 x1 XOR x2 XOR x3 XOR ... XOR xn 的例子，若用樹將ouput結果劃出，則這棵樹可能有log(n)層，但有一個shallow neural network存在，假設僅一層hidden layer，則這一層需要有很多參數來窮舉所有可能性。


## Forward and backward functions

![](https://i.imgur.com/ZdQC2gV.png)


不管總共有幾層的神經網路，我們只將注意力集中在某一層，假設這一層是第 l 層。

![](https://i.imgur.com/9VAvGAn.png)

![](https://i.imgur.com/i4DlQqH.png)

```
下圖是Andrew的參數調整公式

除了內部有蠻多小錯誤被我直接塗白外，跟上述的公式最大的差異在：
我認為1/m應該被包在最頂層的dZ中
因此後續的所有變動量，都繼承了最開始的1/m，便不再多除一次了
```

![](https://i.imgur.com/dpejPXd.png)


## Parameters v.s. Hyperparameters

### Hyperparameters

所有要告訴你的learning algorithm的參數如下都是Hyperparameters，例如：

- W, b, ...

- learing rate alpha

- hidden layer L

- hidden unit: n1, n2, n3, ...

- choice of activation functions

知道超參數的最佳值是很難的，往往都是透過反覆實驗調整而來

對一個新的題目，如何系統化的選擇超參數

- 例如先測試知道哪些參數範圍是可行的

- 例如有一些其他地方借鑑的經驗法則
