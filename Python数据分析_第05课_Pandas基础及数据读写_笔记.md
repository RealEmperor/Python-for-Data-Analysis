
# pandas

## Series



```python
import numpy as np
import pandas as pd
import sys
from pandas import Series, DataFrame


obj = Series([4, 7, -5, 3])
print(obj)
print(obj.values)
print(obj.index)
```

    0    4
    1    7
    2   -5
    3    3
    dtype: int64
    [ 4  7 -5  3]
    RangeIndex(start=0, stop=4, step=1)
    


```python
obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print(obj2)
print(obj2.index)
print(obj2['a'])
```

    d    4
    b    7
    a   -5
    c    3
    dtype: int64
    Index(['d', 'b', 'a', 'c'], dtype='object')
    -5
    


```python
obj2['d'] = 6
obj2[['c', 'a', 'd']]
```




    c    3
    a   -5
    d    6
    dtype: int64




```python
obj2[obj2 > 0]
```




    d    6
    b    7
    c    3
    dtype: int64




```python
obj2 * 2
```




    d    12
    b    14
    a   -10
    c     6
    dtype: int64




```python
np.exp(obj2)
```




    d     403.428793
    b    1096.633158
    a       0.006738
    c      20.085537
    dtype: float64




```python
'b' in obj2
```




    True




```python
'e' in obj2
```




    False




```python
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
print(obj3)
```

    Ohio      35000
    Oregon    16000
    Texas     71000
    Utah       5000
    dtype: int64
    


```python
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
print(obj4)
```

    California        NaN
    Ohio          35000.0
    Oregon        16000.0
    Texas         71000.0
    dtype: float64
    


```python
pd.isnull(obj4)
```




    California     True
    Ohio          False
    Oregon        False
    Texas         False
    dtype: bool




```python
pd.notnull(obj4)
```




    California    False
    Ohio           True
    Oregon         True
    Texas          True
    dtype: bool




```python
obj4.isnull()
```




    California     True
    Ohio          False
    Oregon        False
    Texas         False
    dtype: bool




```python
print(obj3)
print(obj4)
print(obj3 + obj4)
```

    Ohio      35000
    Oregon    16000
    Texas     71000
    Utah       5000
    dtype: int64
    California        NaN
    Ohio          35000.0
    Oregon        16000.0
    Texas         71000.0
    dtype: float64
    California         NaN
    Ohio           70000.0
    Oregon         32000.0
    Texas         142000.0
    Utah               NaN
    dtype: float64
    


```python
obj4.name = 'population'
obj4.index.name = 'state'
print(obj4)
```

    state
    California        NaN
    Ohio          35000.0
    Oregon        16000.0
    Texas         71000.0
    Name: population, dtype: float64
    


```python
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
print(obj)
```

    Bob      4
    Steve    7
    Jeff    -5
    Ryan     3
    dtype: int64
    

## dataframe


```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)

print(frame)
```

       pop   state  year
    0  1.5    Ohio  2000
    1  1.7    Ohio  2001
    2  3.6    Ohio  2002
    3  2.4  Nevada  2001
    4  2.9  Nevada  2002
    


```python
DataFrame(data, columns=['year', 'state', 'pop'])
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
      <th>year</th>
      <th>state</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])
print(frame2)
print(frame2.columns)
```

           year   state  pop debt
    one    2000    Ohio  1.5  NaN
    two    2001    Ohio  1.7  NaN
    three  2002    Ohio  3.6  NaN
    four   2001  Nevada  2.4  NaN
    five   2002  Nevada  2.9  NaN
    Index(['year', 'state', 'pop', 'debt'], dtype='object')
    


```python
print(frame2['state'])
print(frame2.year)
print(frame2.loc['three'])
```

    one        Ohio
    two        Ohio
    three      Ohio
    four     Nevada
    five     Nevada
    Name: state, dtype: object
    one      2000
    two      2001
    three    2002
    four     2001
    five     2002
    Name: year, dtype: int64
    year     2002
    state    Ohio
    pop       3.6
    debt      NaN
    Name: three, dtype: object
    


```python
frame2['debt'] = 16.5
print(frame2)
```

           year   state  pop  debt
    one    2000    Ohio  1.5  16.5
    two    2001    Ohio  1.7  16.5
    three  2002    Ohio  3.6  16.5
    four   2001  Nevada  2.4  16.5
    five   2002  Nevada  2.9  16.5
    


```python
frame2['debt'] = np.arange(5.)
print(frame2)
```

           year   state  pop  debt
    one    2000    Ohio  1.5   0.0
    two    2001    Ohio  1.7   1.0
    three  2002    Ohio  3.6   2.0
    four   2001  Nevada  2.4   3.0
    five   2002  Nevada  2.9   4.0
    


```python
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
print(frame2)
```

           year   state  pop  debt
    one    2000    Ohio  1.5   NaN
    two    2001    Ohio  1.7  -1.2
    three  2002    Ohio  3.6   NaN
    four   2001  Nevada  2.4  -1.5
    five   2002  Nevada  2.9  -1.7
    


```python
frame2['eastern'] = frame2.state == 'Ohio'
print(frame2)
```

           year   state  pop  debt  eastern
    one    2000    Ohio  1.5   NaN     True
    two    2001    Ohio  1.7  -1.2     True
    three  2002    Ohio  3.6   NaN     True
    four   2001  Nevada  2.4  -1.5    False
    five   2002  Nevada  2.9  -1.7    False
    


```python
del frame2['eastern']
print(frame2.columns)
```

    Index(['year', 'state', 'pop', 'debt'], dtype='object')
    


```python
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
print(frame3)
print(frame3.T)
```

          Nevada  Ohio
    2000     NaN   1.5
    2001     2.4   1.7
    2002     2.9   3.6
            2000  2001  2002
    Nevada   NaN   2.4   2.9
    Ohio     1.5   1.7   3.6
    


```python
DataFrame(pop, index=[2001, 2002, 2003])
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
      <th>Nevada</th>
      <th>Ohio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>2.4</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>2.9</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)
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
      <th>Nevada</th>
      <th>Ohio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000</th>
      <td>NaN</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>2.4</td>
      <td>1.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame3.index.name = 'year'
frame3.columns.name = 'state'
print(frame3)
print(frame3.values)
```

    state  Nevada  Ohio
    year               
    2000      NaN   1.5
    2001      2.4   1.7
    2002      2.9   3.6
    [[ nan  1.5]
     [ 2.4  1.7]
     [ 2.9  3.6]]
    

## 索引对象


```python
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
print(index)
print(index[1:])

# index[1] = 'd'
# TypeError: Index does not support mutable operations
```

    Index(['a', 'b', 'c'], dtype='object')
    Index(['b', 'c'], dtype='object')
    


```python
index = pd.Index(np.arange(3))
obj2 = Series([1.5, -2.5, 0], index=index)
obj2.index is index
```




    True




```python
print(frame3)
print('Ohio' in frame3.columns)
print(2003 in frame3.index)
```

    state  Nevada  Ohio
    year               
    2000      NaN   1.5
    2001      2.4   1.7
    2002      2.9   3.6
    True
    False
    

## 数据读取

### 读取文本格式数据 pd.read_csv


```python
# 读取文本格式数据
path = 'data/'
df = pd.read_csv(path + '/ex1.csv')
print(df)
```

       a   b   c   d message
    0  1   2   3   4   hello
    1  5   6   7   8   world
    2  9  10  11  12     foo
    

### OSError: Initializing from file failed 处理方法

出现这个问题大部分原因是文件路径中包含中文，解决办法有两个

1.参数中添加 engine='python' 

data = pd.read_csv("E:\Python数据分析\data\ex1.csv",engine='python')

2.通过 open 读取

data = pd.read_csv(open("E:\Python数据分析\data\ex1.csv"))



```python
pd.read_table(path + '/ex1.csv', sep=',')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv(path + '/ex2.csv', header=None)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv(path + '/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv(path + '/ex2.csv', names=names, index_col='message')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
    <tr>
      <th>message</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hello</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>world</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>foo</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
parsed = pd.read_csv(path + '/csv_mindex.csv', index_col=['key1', 'key2'])
print(parsed)
```

               value1  value2
    key1 key2                
    one  a          1       2
         b          3       4
         c          5       6
         d          7       8
    two  a          9      10
         b         11      12
         c         13      14
         d         15      16
    


```python
list(open(path + '/ex3.txt'))
result = pd.read_table(path + '/ex3.txt', sep='\s+')
print(result)
```

                A         B         C
    aaa -0.264438 -1.026059 -0.619500
    bbb  0.927272  0.302904 -0.032399
    ccc -0.264273 -0.386314 -0.217601
    ddd -0.871858 -0.348382  1.100491
    


```python
pd.read_csv(path + '/ex4.csv', skiprows=[0, 2, 3])
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
result = pd.read_csv(path + '/ex5.csv')
print(result)
print(pd.isnull(result))
```

      something  a   b     c   d message
    0       one  1   2   3.0   4     NaN
    1       two  5   6   NaN   8   world
    2     three  9  10  11.0  12     foo
       something      a      b      c      d  message
    0      False  False  False  False  False     True
    1      False  False  False   True  False    False
    2      False  False  False  False  False    False
    


```python
result = pd.read_csv(path + '/ex5.csv', na_values=['NULL'])
print(result)
```

      something  a   b     c   d message
    0       one  1   2   3.0   4     NaN
    1       two  5   6   NaN   8   world
    2     three  9  10  11.0  12     foo
    


```python
# 把符合条件的数据赋值为NaN
result = pd.read_csv(path + '/ex5.csv', na_values=['NULL'])
print(result)
```

      something  a   b     c   d message
    0       one  1   2   3.0   4     NaN
    1       two  5   6   NaN   8   world
    2     three  9  10  11.0  12     foo
    


```python
sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
pd.read_csv(path + '/ex5.csv', na_values=sentinels)
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 逐行读取文本文件
result = pd.read_csv(path + '/ex6.csv')
print(result)
pd.read_csv(path + '/ex6.csv', nrows=5)
```

               one       two     three      four key
    0     0.467976 -0.038649 -0.295344 -1.824726   L
    1    -0.358893  1.404453  0.704965 -0.200638   B
    2    -0.501840  0.659254 -0.421691 -0.057688   G
    3     0.204886  1.074134  1.388361 -0.982404   R
    4     0.354628 -0.133116  0.283763 -0.837063   Q
    5     1.817480  0.742273  0.419395 -2.251035   Q
    6    -0.776764  0.935518 -0.332872 -1.875641   U
    7    -0.913135  1.530624 -0.572657  0.477252   K
    8     0.358480 -0.497572 -0.367016  0.507702   S
    9    -1.740877 -1.160417 -1.637830  2.172201   G
    10    0.240564 -0.328249  1.252155  1.072796   8
    11    0.764018  1.165476 -0.639544  1.495258   R
    12    0.571035 -0.310537  0.582437 -0.298765   1
    13    2.317658  0.430710 -1.334216  0.199679   P
    14    1.547771 -1.119753 -2.277634  0.329586   J
    15   -1.310608  0.401719 -1.000987  1.156708   E
    16   -0.088496  0.634712  0.153324  0.415335   B
    17   -0.018663 -0.247487 -1.446522  0.750938   A
    18   -0.070127 -1.579097  0.120892  0.671432   F
    19   -0.194678 -0.492039  2.359605  0.319810   H
    20   -0.248618  0.868707 -0.492226 -0.717959   W
    21   -1.091549 -0.867110 -0.647760 -0.832562   C
    22    0.641404 -0.138822 -0.621963 -0.284839   C
    23    1.216408  0.992687  0.165162 -0.069619   V
    24   -0.564474  0.792832  0.747053  0.571675   I
    25    1.759879 -0.515666 -0.230481  1.362317   S
    26    0.126266  0.309281  0.382820 -0.239199   L
    27    1.334360 -0.100152 -0.840731 -0.643967   6
    28   -0.737620  0.278087 -0.053235 -0.950972   J
    29   -1.148486 -0.986292 -0.144963  0.124362   Y
    ...        ...       ...       ...       ...  ..
    9970  0.633495 -0.186524  0.927627  0.143164   4
    9971  0.308636 -0.112857  0.762842 -1.072977   1
    9972 -1.627051 -0.978151  0.154745 -1.229037   Z
    9973  0.314847  0.097989  0.199608  0.955193   P
    9974  1.666907  0.992005  0.496128 -0.686391   S
    9975  0.010603  0.708540 -1.258711  0.226541   K
    9976  0.118693 -0.714455 -0.501342 -0.254764   K
    9977  0.302616 -2.011527 -0.628085  0.768827   H
    9978 -0.098572  1.769086 -0.215027 -0.053076   A
    9979 -0.019058  1.964994  0.738538 -0.883776   F
    9980 -0.595349  0.001781 -1.423355 -1.458477   M
    9981  1.392170 -1.396560 -1.425306 -0.847535   H
    9982 -0.896029 -0.152287  1.924483  0.365184   6
    9983 -2.274642 -0.901874  1.500352  0.996541   N
    9984 -0.301898  1.019906  1.102160  2.624526   I
    9985 -2.548389 -0.585374  1.496201 -0.718815   D
    9986 -0.064588  0.759292 -1.568415 -0.420933   E
    9987 -0.143365 -1.111760 -1.815581  0.435274   2
    9988 -0.070412 -1.055921  0.338017 -0.440763   X
    9989  0.649148  0.994273 -1.384227  0.485120   Q
    9990 -0.370769  0.404356 -1.051628 -1.050899   8
    9991 -0.409980  0.155627 -0.818990  1.277350   W
    9992  0.301214 -1.111203  0.668258  0.671922   A
    9993  1.821117  0.416445  0.173874  0.505118   X
    9994  0.068804  1.322759  0.802346  0.223618   H
    9995  2.311896 -0.417070 -1.409599 -0.515821   L
    9996 -0.479893 -0.650419  0.745152 -0.646038   E
    9997  0.523331  0.787112  0.486066  1.093156   K
    9998 -0.362559  0.598894 -1.843201  0.887292   G
    9999 -0.096376 -1.012999 -0.657431 -0.573315   0
    
    [10000 rows x 5 columns]
    




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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.467976</td>
      <td>-0.038649</td>
      <td>-0.295344</td>
      <td>-1.824726</td>
      <td>L</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.358893</td>
      <td>1.404453</td>
      <td>0.704965</td>
      <td>-0.200638</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.501840</td>
      <td>0.659254</td>
      <td>-0.421691</td>
      <td>-0.057688</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.204886</td>
      <td>1.074134</td>
      <td>1.388361</td>
      <td>-0.982404</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.354628</td>
      <td>-0.133116</td>
      <td>0.283763</td>
      <td>-0.837063</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>




```python
chunker = pd.read_csv(path + '/ex6.csv', chunksize=1000)
tot = Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)

tot = tot.sort_values(ascending=False)
print(tot[:10])
```

    E    368.0
    X    364.0
    L    346.0
    O    343.0
    Q    340.0
    M    338.0
    J    337.0
    F    335.0
    K    334.0
    H    330.0
    dtype: float64
    

## 文件写出 to_csv


```python
# 文件写出
data = pd.read_csv(path + '/ex5.csv')
print(data)
data.to_csv(path + '/out.csv')
```

      something  a   b     c   d message
    0       one  1   2   3.0   4     NaN
    1       two  5   6   NaN   8   world
    2     three  9  10  11.0  12     foo
    


```python
data.to_csv(sys.stdout, sep='|')
```

    |something|a|b|c|d|message
    0|one|1|2|3.0|4|
    1|two|5|6||8|world
    2|three|9|10|11.0|12|foo
    


```python
data.to_csv(sys.stdout, na_rep='NULL')
```

    ,something,a,b,c,d,message
    0,one,1,2,3.0,4,NULL
    1,two,5,6,NULL,8,world
    2,three,9,10,11.0,12,foo
    


```python
data.to_csv(sys.stdout, index=False, header=False)
```

    one,1,2,3.0,4,
    two,5,6,,8,world
    three,9,10,11.0,12,foo
    


```python
data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])
```

    a,b,c
    1,2,3.0
    5,6,
    9,10,11.0
    


```python
dates = pd.date_range('1/1/2000', periods=7)
ts = Series(np.arange(7), index=dates)
ts.to_csv('data/tseries.csv')

Series.from_csv('data/tseries.csv', parse_dates=True)
```




    2000-01-01    0
    2000-01-02    1
    2000-01-03    2
    2000-01-04    3
    2000-01-05    4
    2000-01-06    5
    2000-01-07    6
    dtype: int64



手工处理分隔符格式 csv


```python
# 手工处理分隔符格式
import csv

f = open(path + '/ex7.csv')

reader = csv.reader(f)

for line in reader:
    print(line)
```

    ['a', 'b', 'c']
    ['1', '2', '3']
    ['1', '2', '3', '4']
    


```python
lines = list(csv.reader(open(path + '/ex7.csv')))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}
print(data_dict)
```

    {'a': ('1', '1'), 'b': ('2', '2'), 'c': ('3', '3')}
    


```python
class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL


with open(path + 'mydata.csv', 'w') as f:
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))
pd.read_table(path + 'mydata.csv', sep=';')
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



# Excel数据

## 生成xls工作薄


```python
# Excel数据
# 生成xls工作薄
import xlrd, xlwt

path = 'data/'

wb = xlwt.Workbook()
print(wb)

wb.add_sheet('first_sheet', cell_overwrite_ok=True)
wb.get_active_sheet()

ws_1 = wb.get_sheet(0)
print(ws_1)

ws_2 = wb.add_sheet('second_sheet')

data = np.arange(1, 65).reshape((8, 8))
print(data)

ws_1.write(0, 0, 100)

for c in range(data.shape[0]):
    for r in range(data.shape[1]):
        ws_1.write(r, c, int(data[c, r]))
        ws_2.write(r, c, int(data[r, c]))

wb.save(path + 'workbook.xls')

```

    <xlwt.Workbook.Workbook object at 0x00000194ABF89EF0>
    <xlwt.Worksheet.Worksheet object at 0x00000194ABF8CB00>
    [[ 1  2  3  4  5  6  7  8]
     [ 9 10 11 12 13 14 15 16]
     [17 18 19 20 21 22 23 24]
     [25 26 27 28 29 30 31 32]
     [33 34 35 36 37 38 39 40]
     [41 42 43 44 45 46 47 48]
     [49 50 51 52 53 54 55 56]
     [57 58 59 60 61 62 63 64]]
    


```python
# 从工作薄中读取
book = xlrd.open_workbook(path + 'workbook.xls')
book.sheet_names()

sheet_1 = book.sheet_by_name('first_sheet')
sheet_2 = book.sheet_by_index(1)
print(sheet_1)
print(sheet_2.name)

print(sheet_1.ncols)
print(sheet_1.nrows)

cl = sheet_1.cell(0, 0)
print(cl.value)
print(cl.ctype)

print(sheet_2.row(3))
print(sheet_2.col(3))

print(sheet_1.col_values(3, start_rowx=3, end_rowx=7))
print(sheet_1.row_values(3, start_colx=3, end_colx=7))

for c in range(sheet_1.ncols):
    for r in range(sheet_1.nrows):
        print('%s' % sheet_1.cell(r, c).value)
```

## 生成xlsx工作薄

### 从工作薄中读取

### 使用pandas读取


```python
# 使用pandas读取
xls_file = pd.ExcelFile(path + 'workbook.xls')
table = xls_file.parse('first_sheet')
```

# JSON数据


```python
# JSON数据
import json

obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
              {"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""

result = json.loads(obj)
print(result)

asjson = json.dumps(result)

siblings = DataFrame(result['siblings'], columns=['name', 'age'])
print(siblings)
```

    {'name': 'Wes', 'places_lived': ['United States', 'Spain', 'Germany'], 'pet': None, 'siblings': [{'name': 'Scott', 'age': 25, 'pet': 'Zuko'}, {'name': 'Katie', 'age': 33, 'pet': 'Cisco'}]}
        name  age
    0  Scott   25
    1  Katie   33
    

# 二进制数据格式


```python
# 二进制数据格式
# pickle
frame = pd.read_csv(path + '/ex1.csv')
print(frame)
frame.to_pickle(path + '/frame_pickle')

pd.read_pickle(path + '/frame_pickle')

```

       a   b   c   d message
    0  1   2   3   4   hello
    1  5   6   7   8   world
    2  9  10  11  12     foo
    




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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>



# HDF5格式


```python
# HDF5格式
store = pd.HDFStore('mydata.h5')
store['obj1'] = frame
store['obj1_col'] = frame['a']
print(store)

print(store['obj1'])

store.close()
```

    <class 'pandas.io.pytables.HDFStore'>
    File path: mydata.h5
    /obj1                frame        (shape->[3,5])
    /obj1_col            series       (shape->[3])  
       a   b   c   d message
    0  1   2   3   4   hello
    1  5   6   7   8   world
    2  9  10  11  12     foo
    


```python
import os

os.remove('mydata.h5')
```

# 使用HTML和Web API


```python
# 使用HTML和Web API
import requests

url = 'https://api.github.com/repos/pydata/pandas/milestones/28/labels'
resp = requests.get(url)
print(resp)

data = json.loads(resp.text)

issue_labels = DataFrame(data)
print(issue_labels)
```

    <Response [200]>
         color  default        id             name                   node_id  \
    0   e10c02    False     76811              Bug      MDU6TGFiZWw3NjgxMQ==   
    1   4E9A06    False     76812      Enhancement      MDU6TGFiZWw3NjgxMg==   
    2   FCE94F    False    127681         Refactor      MDU6TGFiZWwxMjc2ODE=   
    3   75507B    False    129350            Build      MDU6TGFiZWwxMjkzNTA=   
    4   3465A4    False    134699             Docs      MDU6TGFiZWwxMzQ2OTk=   
    5   AFEEEE    False    211840       Timeseries      MDU6TGFiZWwyMTE4NDA=   
    6   729FCF    False    233160          Groupby      MDU6TGFiZWwyMzMxNjA=   
    7   06909A    False   2301354          Data IO      MDU6TGFiZWwyMzAxMzU0   
    8   8AE234    False   2413328    Visualization      MDU6TGFiZWwyNDEzMzI4   
    9   0b02e1    False   2822098         Indexing      MDU6TGFiZWwyODIyMDk4   
    10  d7e102    False   2822342     Missing-data      MDU6TGFiZWwyODIyMzQy   
    11  a10c02    False   8935311      Performance      MDU6TGFiZWw4OTM1MzEx   
    12  02d7e1    False  13098779        Reshaping  MDU6TGFiZWwxMzA5ODc3OQ==   
    13  e102d8    False  31404521           Dtypes  MDU6TGFiZWwzMTQwNDUyMQ==   
    14  DDDDDD    False  32933285            Admin  MDU6TGFiZWwzMjkzMzI4NQ==   
    15  AD7FA8    False  35818298       API Design  MDU6TGFiZWwzNTgxODI5OA==   
    16  ffa0ff    False  42670965  Error Reporting  MDU6TGFiZWw0MjY3MDk2NQ==   
    17  006b75    False  47223669          Numeric  MDU6TGFiZWw0NzIyMzY2OQ==   
    18  5319e7    False  47229171           IO CSV  MDU6TGFiZWw0NzIyOTE3MQ==   
    19  5319e7    False  47232590           IO SQL  MDU6TGFiZWw0NzIzMjU5MA==   
    20  a2bca7    False  48070600               CI  MDU6TGFiZWw0ODA3MDYwMA==   
    21  fbca04    False  49094459        Internals  MDU6TGFiZWw0OTA5NDQ1OQ==   
    22  5319e7    False  49597148        Timedelta  MDU6TGFiZWw0OTU5NzE0OA==   
    23  0052cc    False  53181044        Frequency  MDU6TGFiZWw1MzE4MTA0NA==   
    24  fbca04    False  57186974          Windows  MDU6TGFiZWw1NzE4Njk3NA==   
    25  eb6420    False  57296398            Algos  MDU6TGFiZWw1NzI5NjM5OA==   
    26  5319e7    False  60458168        Timezones  MDU6TGFiZWw2MDQ1ODE2OA==   
    27  eb6420    False  60635328           Period  MDU6TGFiZWw2MDYzNTMyOA==   
    28  207de5    False  71268330       MultiIndex  MDU6TGFiZWw3MTI2ODMzMA==   
    29  e11d21    False  78527356      Categorical  MDU6TGFiZWw3ODUyNzM1Ng==   
    
                                                      url  
    0   https://api.github.com/repos/pandas-dev/pandas...  
    1   https://api.github.com/repos/pandas-dev/pandas...  
    2   https://api.github.com/repos/pandas-dev/pandas...  
    3   https://api.github.com/repos/pandas-dev/pandas...  
    4   https://api.github.com/repos/pandas-dev/pandas...  
    5   https://api.github.com/repos/pandas-dev/pandas...  
    6   https://api.github.com/repos/pandas-dev/pandas...  
    7   https://api.github.com/repos/pandas-dev/pandas...  
    8   https://api.github.com/repos/pandas-dev/pandas...  
    9   https://api.github.com/repos/pandas-dev/pandas...  
    10  https://api.github.com/repos/pandas-dev/pandas...  
    11  https://api.github.com/repos/pandas-dev/pandas...  
    12  https://api.github.com/repos/pandas-dev/pandas...  
    13  https://api.github.com/repos/pandas-dev/pandas...  
    14  https://api.github.com/repos/pandas-dev/pandas...  
    15  https://api.github.com/repos/pandas-dev/pandas...  
    16  https://api.github.com/repos/pandas-dev/pandas...  
    17  https://api.github.com/repos/pandas-dev/pandas...  
    18  https://api.github.com/repos/pandas-dev/pandas...  
    19  https://api.github.com/repos/pandas-dev/pandas...  
    20  https://api.github.com/repos/pandas-dev/pandas...  
    21  https://api.github.com/repos/pandas-dev/pandas...  
    22  https://api.github.com/repos/pandas-dev/pandas...  
    23  https://api.github.com/repos/pandas-dev/pandas...  
    24  https://api.github.com/repos/pandas-dev/pandas...  
    25  https://api.github.com/repos/pandas-dev/pandas...  
    26  https://api.github.com/repos/pandas-dev/pandas...  
    27  https://api.github.com/repos/pandas-dev/pandas...  
    28  https://api.github.com/repos/pandas-dev/pandas...  
    29  https://api.github.com/repos/pandas-dev/pandas...  
    

# 使用数据库


```python
# 使用数据库
import sqlite3

query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
 c REAL,        d INTEGER
);"""

con = sqlite3.connect(':memory:')
con.execute(query)
con.commit()
```


```python
data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"

con.executemany(stmt, data)
con.commit()
```


```python
cursor = con.execute('select * from test')
rows = cursor.fetchall()
print(rows)
```

    [('Atlanta', 'Georgia', 1.25, 6), ('Tallahassee', 'Florida', 2.6, 3), ('Sacramento', 'California', 1.7, 5)]
    


```python
print(cursor.description)
```

    (('a', None, None, None, None, None, None), ('b', None, None, None, None, None, None), ('c', None, None, None, None, None, None), ('d', None, None, None, None, None, None))
    


```python
DataFrame(rows, columns=list(zip(*cursor.description))[0])
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Atlanta</td>
      <td>Georgia</td>
      <td>1.25</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tallahassee</td>
      <td>Florida</td>
      <td>2.60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sacramento</td>
      <td>California</td>
      <td>1.70</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pandas.io.sql as sql

sql.read_sql('select * from test', con)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Atlanta</td>
      <td>Georgia</td>
      <td>1.25</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tallahassee</td>
      <td>Florida</td>
      <td>2.60</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sacramento</td>
      <td>California</td>
      <td>1.70</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



参考资料：炼数成金Python数据分析课程
