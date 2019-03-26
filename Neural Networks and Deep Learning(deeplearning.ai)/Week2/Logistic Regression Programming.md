# Logistic Regression Programming


**What you need to remember:**

Common steps for pre-processing a new dataset are:
- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
    ```python
    train_set.shape
    ```
    
- Reshape the datasets such that each example is now a vector of size (num_px \* num_px \* 3, 1)
    ```python
    # 注意有轉置，為了讓矩陣變成 (k * n)，n為樣本數，k為feature數
    train_set_flatten = train_set.reshape(train_set.shape[0], -1).T
    ```

- "Standardize" the data
    ```python
    train_set_x = train_set_flatten / 255.
    #train_set_x = train_set_flatten / 255.0
    ```
