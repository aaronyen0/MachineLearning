# Incremental Quality Infernce in Crowdsourcing (INQUIRE)
    Jianhong Feng, Guoliang Li, Henan Wang, and Jianhua Feng
    Department of Computer Science, Tsinghua University, Beijing 100084, China
    

## **名詞解釋**
![](https://i.imgur.com/3waLyay.png)


## **INQUIRE Framework**

![](https://i.imgur.com/6GVCsKY.png)

- **Algorithm**

    ![](https://i.imgur.com/YQbNxZv.png)


## **Model**
![](https://i.imgur.com/6aIP4Cr.png)

  
## **Updating Models**

### Updating Question Model

論文提出四種方式去更新Question Model，其中兩個是Weighted Strategy(WS)的方法，兩個是Probability Strategy(PS)的方法，目前預計採用PS-Gamma。

![](https://i.imgur.com/CT9LRch.png)


### Updating Worker Model

INQUIRE has to compute the change of workers’ accuracy in time, and update g workers’ model when a question has received g answers.

![](https://i.imgur.com/H0OEBS7.png)
