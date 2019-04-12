# **Random Task Generator**

## **module : random_task**

- 使用方法
    - 初始化class: RandTask()
        - 外部定義每個任務的相對頻率
        - 在初始化引數中以list的形式填入相對頻率

    - 呼叫function: get_random_task_idx()
        - 回傳一個任務序號供外部使用
        - 序號是根據各自的相對頻率隨機挑選的

