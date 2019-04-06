### 语料库的加载

由于语料库过大超过100M，不能commit到online，所以给出链接

https://pan.baidu.com/s/1FHl_bQkYucvVk-j2KG4dxA

### cnn 模型

* 参数选择：

  feature_map = 32

  window_size=[3,4,5]

  mini_batch=20

  vocabulary_size=100

  dropout_keep_pro=0.5

  learning_rate=1e-3

* 备注

  由于host gpu有限，所以不能fully跑数据集，只选取了demo大小的数据集进行跑。

* 其他

  train,test按照4:1互斥划分

### 结果

- ![avatar](/image/cnn_result.png)