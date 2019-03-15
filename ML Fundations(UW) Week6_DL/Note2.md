# Deep Learning score card

## Pros
- Enables learning of features rather than hand tuning
- Impressive performance gains
  - Computer vision
  - Speech recognition
  - Some tet analysis
- Potential for more impact


## Cons
- Requires a lot of data for high accuracy
- Computationally really expensive (參數過多，在做架構調整時，難以觀察哪些參數是有用或無用的)
- Extremely hard to tune
  - Choice of architecture
  - Parameter types
  - Hyperparameters
  - Learning algorithm
  - ...
  
  
## Deep Features
- DL模型參數眾多，訓練時有極高的資料量要求，若是收集不到足夠的數據，便無法很好的訓練這些參數，但其實還是有方法降低對數據量的需求，其中一個就是Transfer learning。
- Transfer learning : Use data from one task to help learn on another
  - 例如：我今天有很大量的貓狗數據，並用這些數據訓練好一個深度學習模型，取得各種特徵。今天來了另一個任務在辨識桌子，但是資料量相對很少。是否能用貓狗模型的特徵，來當作桌子模型的起始特徵，並繼續訓練呢？或是能不能在貓狗模型中新增一個桌子類別？
  
    ![](https://github.com/worcdlo/Machine-Learning/blob/master/ML%20Fundations(UW)%20Week6_DL/note1.GIF)
       ```
       在很多的研究中都曾表明，神經網路在不同深度各有不同用途，
       通常越淺層的部分，擷取得特徵偏向材質、邊角等較通用型態，
       而越靠近底層的參數，擷取的特徵偏向個別類別獨有的。
       因此在使用Transfer learning時，我們常常會運用這個特性，
       只替換最底層的特徵，留下前面訓練好的數據
       就可以大幅的降低訓練需求
       ```

