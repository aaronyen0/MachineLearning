# Machine Learning Foundations: A Case Study Approach(UW)

## Summary


### Lifecycle of ML Production
  - Deployment System (a-b:batch offline setting, c-d:Real-time predictions)
    1. 歷史數據
    2. 用數據來訓練模型
    3. 在雲上面部屬
    4. 使用者取得推薦或是根據使用經驗回饋
    
  - Management
    1. When to update a model?
        - Why update? Trend nad user tastes change over time. Model performance drops.
        - Track statostocs pf data over time
        - Monitor both offline & online metrics
        - Update when offline metric diverges form online metrics or not achieving desired targets
      
    2. How to choose between existing models?
        - A/B Testing: Choosing between ML models:給兩組人是用兩種模型，看看兩組人的結果再決定
        - Versioning
        - Provenance
        - Dashboards
        - Reports
    3. Copntinuous evaluation and testing
    
  - Evaluation / Monitoring
    1. Evaluation = Predictions + Metric
    2. 我們從用或中蒐集什麼數據?
    3. 用戶的即時行為是什麼?
    4. ML是否真的很好且恰當的在運作?


### Open Challenges
  - Model selection
  - How do we represent the data? Feature engineering/representation
    1. Normalize?
    2. TF-IDF?
    3. Bag of word raw counts?
    4. Bigrams?
    5. Trigrams?
    6. ...
  - Scaling
    1. Data is getting big (devices, online stores, youtube, wiki, health record, ...)
    2. model are getting big, and more complication
    3. Exponentially increasing rate of increased speed for our processors. But that stopped about a decade ago.
        - parallel, GPU, clouds, Multicore, ...
        - But scalable ML in these systems is hard, especially in terms of:
          - Programmablility
          - Data distribution
          - Failures
