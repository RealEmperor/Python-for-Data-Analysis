
# 分类算法案例：预测饭店销量

## 导入数据


```python
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

inputfile = 'data/sales_data.xls'
data = pd.read_excel(inputfile, index_col=u'序号')  # 导入数据
data
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>天气</th>
      <th>是否周末</th>
      <th>是否有促销</th>
      <th>销量</th>
    </tr>
    <tr>
      <th>序号</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>坏</td>
      <td>是</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>2</th>
      <td>坏</td>
      <td>是</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>3</th>
      <td>坏</td>
      <td>是</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>4</th>
      <td>坏</td>
      <td>否</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>5</th>
      <td>坏</td>
      <td>是</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>6</th>
      <td>坏</td>
      <td>否</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>7</th>
      <td>坏</td>
      <td>是</td>
      <td>否</td>
      <td>高</td>
    </tr>
    <tr>
      <th>8</th>
      <td>好</td>
      <td>是</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>9</th>
      <td>好</td>
      <td>是</td>
      <td>否</td>
      <td>高</td>
    </tr>
    <tr>
      <th>10</th>
      <td>好</td>
      <td>是</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>11</th>
      <td>好</td>
      <td>是</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>12</th>
      <td>好</td>
      <td>是</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>13</th>
      <td>好</td>
      <td>是</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>14</th>
      <td>坏</td>
      <td>是</td>
      <td>是</td>
      <td>低</td>
    </tr>
    <tr>
      <th>15</th>
      <td>好</td>
      <td>否</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>16</th>
      <td>好</td>
      <td>否</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>17</th>
      <td>好</td>
      <td>否</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>18</th>
      <td>好</td>
      <td>否</td>
      <td>是</td>
      <td>高</td>
    </tr>
    <tr>
      <th>19</th>
      <td>好</td>
      <td>否</td>
      <td>否</td>
      <td>高</td>
    </tr>
    <tr>
      <th>20</th>
      <td>坏</td>
      <td>否</td>
      <td>否</td>
      <td>低</td>
    </tr>
    <tr>
      <th>21</th>
      <td>坏</td>
      <td>否</td>
      <td>是</td>
      <td>低</td>
    </tr>
    <tr>
      <th>22</th>
      <td>坏</td>
      <td>否</td>
      <td>是</td>
      <td>低</td>
    </tr>
    <tr>
      <th>23</th>
      <td>坏</td>
      <td>否</td>
      <td>是</td>
      <td>低</td>
    </tr>
    <tr>
      <th>24</th>
      <td>坏</td>
      <td>否</td>
      <td>否</td>
      <td>低</td>
    </tr>
    <tr>
      <th>25</th>
      <td>坏</td>
      <td>是</td>
      <td>否</td>
      <td>低</td>
    </tr>
    <tr>
      <th>26</th>
      <td>好</td>
      <td>否</td>
      <td>是</td>
      <td>低</td>
    </tr>
    <tr>
      <th>27</th>
      <td>好</td>
      <td>否</td>
      <td>是</td>
      <td>低</td>
    </tr>
    <tr>
      <th>28</th>
      <td>坏</td>
      <td>否</td>
      <td>否</td>
      <td>低</td>
    </tr>
    <tr>
      <th>29</th>
      <td>坏</td>
      <td>否</td>
      <td>否</td>
      <td>低</td>
    </tr>
    <tr>
      <th>30</th>
      <td>好</td>
      <td>否</td>
      <td>否</td>
      <td>低</td>
    </tr>
    <tr>
      <th>31</th>
      <td>坏</td>
      <td>是</td>
      <td>否</td>
      <td>低</td>
    </tr>
    <tr>
      <th>32</th>
      <td>好</td>
      <td>否</td>
      <td>是</td>
      <td>低</td>
    </tr>
    <tr>
      <th>33</th>
      <td>好</td>
      <td>否</td>
      <td>否</td>
      <td>低</td>
    </tr>
    <tr>
      <th>34</th>
      <td>好</td>
      <td>否</td>
      <td>否</td>
      <td>低</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 数据是类别标签，要将它转换为数据
# 用1来表示“好”、“是”、“高”这三个属性，用-1来表示“坏”、“否”、“低”
data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data != 1] = -1
x = data.iloc[:, :3].as_matrix().astype(int)
print(x)
y = data.iloc[:, 3].as_matrix().astype(int)
print(y)
```

    [[-1  1  1]
     [-1  1  1]
     [-1  1  1]
     [-1 -1  1]
     [-1  1  1]
     [-1 -1  1]
     [-1  1 -1]
     [ 1  1  1]
     [ 1  1 -1]
     [ 1  1  1]
     [ 1  1  1]
     [ 1  1  1]
     [ 1  1  1]
     [-1  1  1]
     [ 1 -1  1]
     [ 1 -1  1]
     [ 1 -1  1]
     [ 1 -1  1]
     [ 1 -1 -1]
     [-1 -1 -1]
     [-1 -1  1]
     [-1 -1  1]
     [-1 -1  1]
     [-1 -1 -1]
     [-1  1 -1]
     [ 1 -1  1]
     [ 1 -1  1]
     [-1 -1 -1]
     [-1 -1 -1]
     [ 1 -1 -1]
     [-1  1 -1]
     [ 1 -1  1]
     [ 1 -1 -1]
     [ 1 -1 -1]]
    [ 1  1  1  1  1  1  1  1  1  1  1  1  1 -1  1  1  1  1  1 -1 -1 -1 -1 -1 -1
     -1 -1 -1 -1 -1 -1 -1 -1 -1]
    

## 拆分训练数据与测试数据


```python
# 拆分训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```

# 训练KNN分类器


```python
clf = KNeighborsClassifier(algorithm='kd_tree')
clf.fit(x_train, y_train)
```




    KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')



## 测试结果


```python
# 测试结果
answer = clf.predict(x_test)
print(x_test)
print(answer)
print(y_test)
print(np.mean(answer == y_test))
```

    [[ 1 -1  1]
     [-1  1  1]
     [-1 -1  1]
     [ 1 -1 -1]
     [-1  1 -1]
     [ 1 -1  1]
     [ 1  1  1]]
    [-1  1 -1 -1 -1 -1  1]
    [ 1  1  1  1 -1  1  1]
    0.428571428571
    

## 准确率


```python
# 准确率
precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
print(classification_report(y_test, answer, target_names=['高', '低']))
```

                 precision    recall  f1-score   support
    
              高       0.20      1.00      0.33         1
              低       1.00      0.33      0.50         6
    
    avg / total       0.89      0.43      0.48         7
    
    

# 训练贝叶斯分类器


```python
# 训练贝叶斯分类器
clf = BernoulliNB()
clf.fit(x_train, y_train)
```




    BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)



## 测试结果


```python
# 测试结果
answer = clf.predict(x_test)
print(x_test)
print(answer)
print(y_test)
print(np.mean(answer == y_test))
```

    [[ 1 -1  1]
     [-1  1  1]
     [-1 -1  1]
     [ 1 -1 -1]
     [-1  1 -1]
     [ 1 -1  1]
     [ 1  1  1]]
    [-1  1 -1 -1 -1 -1  1]
    [ 1  1  1  1 -1  1  1]
    0.428571428571
    

## 准确率


```python
print(classification_report(y_test, answer, target_names=['低', '高']))
```

                 precision    recall  f1-score   support
    
              低       0.20      1.00      0.33         1
              高       1.00      0.33      0.50         6
    
    avg / total       0.89      0.43      0.48         7
    
    

# 决策树


```python
from sklearn.tree import DecisionTreeClassifier as DTC

dtc = DTC(criterion='entropy')  # 建立决策树模型，基于信息熵
dtc.fit(x_train, y_train)  # 训练模型

# 导入相关函数，可视化决策树。
# 导出的结果是一个dot文件，需要安装Graphviz才能将它转换为pdf或png等格式。

# https://graphviz.gitlab.io/_pages/Download/Download_windows.html
# 安装Graphviz https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.msi

from sklearn.tree import export_graphviz

with open("data/tree.dot", 'w') as f:
    f = export_graphviz(dtc, out_file=f)
```

## 测试结果


```python
# 测试结果
answer = dtc.predict(x_test)
print(x_test)
print(answer)
print(y_test)
print(np.mean(answer == y_test))
```

    [[ 1 -1  1]
     [-1  1  1]
     [-1 -1  1]
     [ 1 -1 -1]
     [-1  1 -1]
     [ 1 -1  1]
     [ 1  1  1]]
    [-1  1 -1 -1 -1 -1  1]
    [ 1  1  1  1 -1  1  1]
    0.428571428571
    

## 准确率


```python
print(classification_report(y_test, answer, target_names=['低', '高']))
```

                 precision    recall  f1-score   support
    
              低       0.20      1.00      0.33         1
              高       1.00      0.33      0.50         6
    
    avg / total       0.89      0.43      0.48         7
    
    

# SVM


```python
from sklearn.svm import SVC

clf = SVC()
clf.fit(x_train, y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)



## 测试结果


```python
# 测试结果
answer = clf.predict(x_test)
print(x_test)
print(answer)
print(y_test)
print(np.mean(answer == y_test))
```

    [[ 1 -1  1]
     [-1  1  1]
     [-1 -1  1]
     [ 1 -1 -1]
     [-1  1 -1]
     [ 1 -1  1]
     [ 1  1  1]]
    [-1  1 -1 -1  1 -1  1]
    [ 1  1  1  1 -1  1  1]
    0.285714285714
    

## 准确率


```python
print(classification_report(y_test, answer, target_names=['低', '高']))
```

                 precision    recall  f1-score   support
    
              低       0.00      0.00      0.00         1
              高       0.67      0.33      0.44         6
    
    avg / total       0.57      0.29      0.38         7
    
    

参考资料：炼数成金Python数据分析课程
