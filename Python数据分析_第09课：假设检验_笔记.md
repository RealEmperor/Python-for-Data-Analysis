# 统计学 

统计学可以分为：描述统计学与推断统计学 

**描述统计学** ：使用特定的数字或图表来体现数据的集中程度和离散程度。例：每次考试算的平均分，最高分，各个分段的人数分布等，也是属于描述统计学的范围。 

**推断统计学** ：根据样本数据推断总体数据特征。例：产品质量检查，一般采用抽检，根 据所抽样本的质量合格率作为总体的质量合格率的一个估计。 

## 描述统计学 
### 数据特征描述分析 

#### 集中趋势 

**均值**：算术平均数，描述平均水平 

```python
from pandas import Series, DataFrame
import numpy as np

arr = [98, 83, 65, 72, 79, 76, 75, 94, 91, 77, 63, 83, 89, 69, 64, 78, 63, 86, 91, 72, 71, 72, 70, 80, 65, 70, 62, 74, 71,
       76]

# 均值
np.mean(arr)
```
    75.966666666666669

```python
np.mean(np.sort(arr)[14:16])
```
    74.5


**中位数**：将数据按大小排列后位于正中间的数，描述中等水平 


```python
# 排序
np.sort(arr)
```
    array([62, 63, 63, 64, 65, 65, 69, 70, 70, 71, 71, 72, 72, 72, 74, 75, 76,
           76, 77, 78, 79, 80, 83, 83, 86, 89, 91, 91, 94, 98])



**众数**：数据中出现最多的数，描述一般水平 
```python
# 求众数的函数
def get_mode(arr):
    mode = []
    arr_appear = dict((a, arr.count(a)) for a in arr)  # 统计各个元素出现的次数
    if max(arr_appear.values()) == 1:  # 如果最大的出现为1  
        return  # 则没有众数
    else:
        for k, v in arr_appear.items():  # 否则，出现次数最大的数字，就是众数  
            if v == max(arr_appear.values()):
                mode.append(k)
    return mode


get_mode(arr)
```
    [72]
&nbsp;  | 优点 | 缺点 
-----------| -----------|------
均值     | 充分利用所有数据，适用性强 | 容易受到极端值影响 
中位数 | 不受极端值影响  | 缺乏敏感性 
众数    |  当数据具有明显的集中趋势时，可能代表性好；不受极端值影响 | 缺乏唯一性：可能有一个，有两个，可能一个都没有 

#### 离散程度的描述 

**极差**：最大值 - 最小值，简单地描述数据的范围大小 

**方差**：在统计学上，更常地是使用方差来描述数据的离散程度——数据离中心越远越离散
$$ \sigma^2= \frac{1}{n}  \sum_{i=1}^n (X_i-\mu)^2  $$ 

其中，𝑋 表示数据集中第 i 个数据的值，$\mu$ 表示数据集的均值 i

**标准差**：$\sigma=\sqrt{\sigma^2}$ ，有效的避免了因单位平方引起的度量问题。标准差的值越大，表示数据越分散。

**偏度**：对数据分布的偏斜程度的衡量 $E[(\frac{X-\mu}{\sigma})^3]$

```python
arr = Series(arr)
# 偏度
arr.skew()
```




    0.57428918733545919

偏度可分为正态分布（偏度=0），右偏分布（也叫正偏分布，其偏度>0），左偏分布（也叫负偏分布，其偏度<0）。

**峰度**：对数据分布的峰态的衡量 $E[(\frac{X-\mu}{\sigma})^4]$

峰度可分为正态分布（峰度值=3），厚尾（峰度值>3），瘦尾（峰度值<3）。
```python
# 峰度
arr.kurt()
```
    -0.42872374795373291
    
**分位数**：是指将一个随机变量的概率分布范围分为几个等份的数值点。

`describe` 返回值包括：数量，均值，标准差，最小值，25%分位数，50%分位数，75%分位数，最大值。

```python
# 描述
arr.describe()
```
    count    30.000000
    mean     75.966667
    std       9.824260
    min      62.000000
    25%      70.000000
    50%      74.500000
    75%      82.250000
    max      98.000000
    dtype: float64
  
#### 多元数据的数据特征    
**协方差**（Covariance）在概率论和统计学中用于衡量两个变量的总体误差。而方差是协方差的一种特殊情况，即当两个变量是相同的情况。

$$   Cov(X,Y)  =E[(X-E[X])(Y-E[Y])] \\
 =E[XY]-E[X]E[Y] 
$$

```python
df = DataFrame({'data1': np.random.randn(5),
                'data2': np.random.randn(5)})
print(df)
# 协方差
df.cov()
```

          data1     data2
    0 -0.481723  1.021694
    1  0.628769 -1.134141
    2 -0.463298 -1.042589
    3  0.159722  0.914932
    4 -0.493772 -0.559481
    




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
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>data1</th>
      <td>0.256699</td>
      <td>-0.100965</td>
    </tr>
    <tr>
      <th>data2</th>
      <td>-0.100965</td>
      <td>1.109845</td>
    </tr>
  </tbody>
</table>
</div>


**相关系数**

$$\rho_{xy}=\frac{Cov(X,Y)}{\sqrt{D(X)}\sqrt{D(Y)}}$$

称为随机变量X和Y的相关系数。

若$\rho_{xy}=0$，则称X与Y不线性相关。
相关系数的绝对值小于等于1 ，$|\rho_{xy}|\leq 1$。越靠近1 相关性越强。

```python
# 相关系数
df.corr()
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
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>data1</th>
      <td>1.000000</td>
      <td>-0.189159</td>
    </tr>
    <tr>
      <th>data2</th>
      <td>-0.189159</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


### 假设检验
假设检验的基本步骤 
1. 提出零假设 
2. 建立检验统计量 
3. 确定否定域/计算p-value 
4. 得出结论 

**零假设**（原假设）是假定一个总体参数等于某个特定值的一个声明，用$H_0$表示。

**备择假设**是假定该总体参数为零假设中假设的值除外的值，用$H_1$表示。

若希望假设的论断成立，设为备择假设；若希望假设的论断不成立，设为零假设。


#### 拒绝域与显著性水平
**拒绝域**：也称否定域，是指检验统计量所有可以拒绝零假设的取值所构成的集合。
**显著性水平**：$\alpha$ 指当零假设正确的时候，检验统计量落在拒绝域的概率。也就是当零假设为真而我们却拒绝零假设这种错误发生的概率。与置信区间中的显著性水平 $\alpha$ 意义一致。常用取值：0.1,0.05,0.01
**临界值**：拒绝域与非拒绝域的分界线
**P-value**：样本发生或者比样本更极端的情况发生的概率，越小越拒绝

#### 决定规则
方法 |拒绝零假设 |不拒绝零假设
--|--|--
临界值法 |检验统计量落在拒绝域 |检验统计量没有落在拒绝域
P-value法 |P-value$\leq\alpha$ |P-value$>\alpha$
另一个选择| 不采用具体的$\alpha$值，写出P-value留给读者自己判断


#### 第一类错误与第二类错误
**第一类错误**：零假设正确的情况下拒绝了零假设，错杀好人
**第二类错误**：零假设错误的情况下没有拒绝零假设，放走坏人

#### 检验统计量（test statistics）
z检验
t检验
卡方检验
F检验

```python
#假设检验
from scipy import stats as ss

df = DataFrame({'data': [10.1, 10, 9.8, 10.5, 9.7, 10.1, 9.9, 10.2, 10.3, 9.9]})
# T-检验
ss.ttest_1samp(a=df, popmean=10)
```
    Ttest_1sampResult(statistic=array([ 0.65465367]), pvalue=array([ 0.52906417]))
一般以P < 0.05 为显著， P <0.01 为非常显著。
P-value 越小越拒绝，上面pvalue=0.52906417，足够大，不能拒绝原假设（均值为10）。

参考资料：炼数成金Python数据分析课程