
# NumPy对向量的处理

向量相加-Python方式


```python
# 向量相加-Python
def pythonsum(n):
    a = list(range(n))
    b = list(range(n))
    c = []
    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i] + b[i])
    return c
```

向量相加-NumPy方式


```python
# 向量相加-NumPy
import numpy


def numpysum(n):
    a = numpy.arange(n) ** 2
    b = numpy.arange(n) ** 3
    c = a + b
    return c
```

## 效率比较


```python
# 效率比较
from datetime import datetime
import numpy as np

size = 1000

start = datetime.now()
c = pythonsum(size)
delta = datetime.now() - start
print("The last 2 elements of the sum", c[-2:])
print("PythonSum elapsed time in microseconds", delta.microseconds)

start = datetime.now()
c = numpysum(size)
delta = datetime.now() - start
print("The last 2 elements of the sum", c[-2:])
print("NumPySum elapsed time in microseconds", delta.microseconds)
```

    The last 2 elements of the sum [995007996, 998001000]
    PythonSum elapsed time in microseconds 997
    The last 2 elements of the sum [995007996 998001000]
    NumPySum elapsed time in microseconds 0
    

## numpy数组


```python
# numpy数组
a = np.arange(5)
a.dtype
```




    dtype('int32')




```python
a
```




    array([0, 1, 2, 3, 4])




```python
a.shape
```




    (5,)



## 创建多维数组


```python
# 创建多维数组
m = np.array([np.arange(2), np.arange(2)])

print(m)

print(m.shape)

print(m.dtype)
```

    [[0 1]
     [0 1]]
    (2, 2)
    int32
    


```python
np.zeros(10)
```




    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])




```python
np.zeros((3, 6))
```




    array([[ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.]])




```python
np.empty((2, 3, 2))
```




    array([[[  1.12319318e-311,   3.16202013e-322],
            [  0.00000000e+000,   0.00000000e+000],
            [  0.00000000e+000,   2.32203106e-056]],
    
           [[  1.39425677e+165,   7.12024499e-067],
            [  1.08628251e-071,   2.19317805e-076],
            [  1.90269056e-052,   1.31378093e-071]]])




```python
np.arange(15)
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])



## 选取数组元素


```python
# 选取数组元素
a = np.array([[1, 2], [3, 4]])

print("In: a")
print(a)

print("In: a[0,0]")
print(a[0, 0])

print("In: a[0,1]")
print(a[0, 1])

print("In: a[1,0]")
print(a[1, 0])

print("In: a[1,1]")
print(a[1, 1])
```

    In: a
    [[1 2]
     [3 4]]
    In: a[0,0]
    1
    In: a[0,1]
    2
    In: a[1,0]
    3
    In: a[1,1]
    4
    

## numpy数据类型


```python
# numpy数据类型
print("In: float64(42)")
print(np.float64(42))
```

    In: float64(42)
    42.0
    


```python
print("In: int8(42.0)")
print(np.int8(42.0))
```

    In: int8(42.0)
    42
    


```python
print("In: bool(42)")
print(np.bool(42))
```

    In: bool(42)
    True
    


```python
print(np.bool(0))
```

    False
    


```python
print("In: bool(42.0)")
print(np.bool(42.0))
```

    In: bool(42.0)
    True
    


```python
print("In: float(True)")
print(np.float(True))
print(np.float(False))
```

    In: float(True)
    1.0
    0.0
    


```python
print("In: arange(7, dtype=uint16)")
print(np.arange(7, dtype=np.uint16))
```

    In: arange(7, dtype=uint16)
    [0 1 2 3 4 5 6]
    


```python
print("In: int(42.0 + 1.j)")
try:
    print(np.int(42.0 + 1.j))
except TypeError:
    print("TypeError")
# Type error
```

    In: int(42.0 + 1.j)
    TypeError
    


```python
print("In: float(42.0 + 1.j)")
print(float(42.0 + 1.j))
# Type error
```

    In: float(42.0 + 1.j)
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-21-5c8b6ddc18ab> in <module>()
          1 print("In: float(42.0 + 1.j)")
    ----> 2 print(float(42.0 + 1.j))
          3 # Type error
    

    TypeError: can't convert complex to float


## 数据类型转换


```python
# 数据类型转换
arr = np.array([1, 2, 3, 4, 5])
arr.dtype
```




    dtype('int32')




```python
float_arr = arr.astype(np.float64)
float_arr.dtype

```




    dtype('float64')




```python
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr
```




    array([  3.7,  -1.2,  -2.6,   0.5,  12.9,  10.1])




```python
arr.astype(np.int32)
```




    array([ 3, -1, -2,  0, 12, 10])




```python
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)
```




    array([  1.25,  -9.6 ,  42.  ])



## 数据类型对象


```python
# 数据类型对象
a = np.array([[1, 2], [3, 4]])

print(a.dtype.byteorder)

print(a.dtype.itemsize)
```

    =
    4
    

## 字符编码


```python
# 字符编码
print(np.arange(7, dtype='f'))
print(np.arange(7, dtype='D'))

print(np.dtype(float))

print(np.dtype('f'))

print(np.dtype('d'))

print(np.dtype('f8'))

print(np.dtype('Float64'))
```

    [ 0.  1.  2.  3.  4.  5.  6.]
    [ 0.+0.j  1.+0.j  2.+0.j  3.+0.j  4.+0.j  5.+0.j  6.+0.j]
    float64
    float32
    float64
    float64
    float64
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:13: DeprecationWarning: Numeric-style type codes are deprecated and will result in an error in the future.
      del sys.path[0]
    

## type类的属性


```python
# dtype类的属性
t = np.dtype('Float64')

print(t.char)

print(t.type)

print(t.str)
```

    d
    <class 'numpy.float64'>
    <f8
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: DeprecationWarning: Numeric-style type codes are deprecated and will result in an error in the future.
      
    

## 创建自定义数据类型


```python
# 创建自定义数据类型
t = np.dtype([('name', np.str_, 40), ('numitems', np.int32), ('price', np.float32)])
print(t)

print(t['name'])

itemz = np.array([('Meaning of life DVD', 42, 3.14), ('Butter', 13, 2.72)], dtype=t)

print(itemz[1])
```

    [('name', '<U40'), ('numitems', '<i4'), ('price', '<f4')]
    <U40
    ('Butter', 13,  2.72000003)
    

## 数组与标量的运算


```python
# 数组与标量的运算
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr
```




    array([[ 1.,  2.,  3.],
           [ 4.,  5.,  6.]])




```python
arr * arr
```




    array([[  1.,   4.,   9.],
           [ 16.,  25.,  36.]])




```python
arr - arr
```




    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])




```python
1 / arr
```




    array([[ 1.        ,  0.5       ,  0.33333333],
           [ 0.25      ,  0.2       ,  0.16666667]])




```python
arr ** 0.5
```




    array([[ 1.        ,  1.41421356,  1.73205081],
           [ 2.        ,  2.23606798,  2.44948974]])



## 一维数组的索引与切片


```python
# 一维数组的索引与切片
a = np.arange(9)

print(a[3:7])

print(a[:7:2])

print(a[::-1])

s = slice(3, 7, 2)
print(a[s])

s = slice(None, None, -1)

print(a[s])
```

    [3 4 5 6]
    [0 2 4 6]
    [8 7 6 5 4 3 2 1 0]
    [3 5]
    [8 7 6 5 4 3 2 1 0]
    

## 多维数组的切片与索引


```python
# 多维数组的切片与索引
b = np.arange(24).reshape(2, 3, 4)

print(b.shape)

print(b)

print(b[0, 0, 0])

print(b[:, 0, 0])

print(b[0])

print(b[0, :, :])

print(b[0, ...])

print(b[0, 1])

print(b[0, 1, ::2])

print(b[..., 1])

print(b[:, 1])

print(b[0, :, 1])

print(b[0, :, -1])

print(b[0, ::-1, -1])

print(b[0, ::2, -1])

print(b[::-1])

s = slice(None, None, -1)
print(b[(s, s, s)])
```

    (2, 3, 4)
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
    
     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]]
    0
    [ 0 12]
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    [4 5 6 7]
    [4 6]
    [[ 1  5  9]
     [13 17 21]]
    [[ 4  5  6  7]
     [16 17 18 19]]
    [1 5 9]
    [ 3  7 11]
    [11  7  3]
    [ 3 11]
    [[[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]
    
     [[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]]
    [[[23 22 21 20]
      [19 18 17 16]
      [15 14 13 12]]
    
     [[11 10  9  8]
      [ 7  6  5  4]
      [ 3  2  1  0]]]
    

## 布尔型索引


```python
# 布尔型索引
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
```


```python
names
```




    array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'], 
          dtype='<U4')




```python
data
```




    array([[-1.01504197, -1.53005352,  0.75853162,  0.35188611],
           [ 0.42691628,  0.64738979, -0.49456879,  0.6420728 ],
           [ 0.79245449, -0.88837584,  1.15706385, -1.40232002],
           [ 0.10355699, -1.15478154,  0.99303669, -1.42431578],
           [-1.08005402, -0.88447402, -0.19597154,  0.50873589],
           [ 1.52942389,  0.37328854, -1.01259458, -0.70578044],
           [-0.50512187,  1.40398143,  0.324043  , -0.58337786]])




```python
names == 'Bob'
```




    array([ True, False, False,  True, False, False, False], dtype=bool)




```python
data[names == 'Bob']
```




    array([[-1.01504197, -1.53005352,  0.75853162,  0.35188611],
           [ 0.10355699, -1.15478154,  0.99303669, -1.42431578]])




```python
data[names == 'Bob', 2:]
```




    array([[ 0.75853162,  0.35188611],
           [ 0.99303669, -1.42431578]])




```python
data[names == 'Bob', 3]
```




    array([ 0.35188611, -1.42431578])




```python
names != 'Bob'
```




    array([False,  True,  True, False,  True,  True,  True], dtype=bool)




```python
data[~(names == 'Bob')]
```




    array([[ 0.42691628,  0.64738979, -0.49456879,  0.6420728 ],
           [ 0.79245449, -0.88837584,  1.15706385, -1.40232002],
           [-1.08005402, -0.88447402, -0.19597154,  0.50873589],
           [ 1.52942389,  0.37328854, -1.01259458, -0.70578044],
           [-0.50512187,  1.40398143,  0.324043  , -0.58337786]])




```python
mask = (names == 'Bob') | (names == 'Will')
mask
```




    array([ True, False,  True,  True,  True, False, False], dtype=bool)




```python
data[mask]
```




    array([[-1.01504197, -1.53005352,  0.75853162,  0.35188611],
           [ 0.79245449, -0.88837584,  1.15706385, -1.40232002],
           [ 0.10355699, -1.15478154,  0.99303669, -1.42431578],
           [-1.08005402, -0.88447402, -0.19597154,  0.50873589]])




```python
data[data < 0] = 0
data
```




    array([[ 0.        ,  0.        ,  0.75853162,  0.35188611],
           [ 0.42691628,  0.64738979,  0.        ,  0.6420728 ],
           [ 0.79245449,  0.        ,  1.15706385,  0.        ],
           [ 0.10355699,  0.        ,  0.99303669,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.50873589],
           [ 1.52942389,  0.37328854,  0.        ,  0.        ],
           [ 0.        ,  1.40398143,  0.324043  ,  0.        ]])




```python
data[names != 'Joe'] = 7
data
```




    array([[ 7.        ,  7.        ,  7.        ,  7.        ],
           [ 0.42691628,  0.64738979,  0.        ,  0.6420728 ],
           [ 7.        ,  7.        ,  7.        ,  7.        ],
           [ 7.        ,  7.        ,  7.        ,  7.        ],
           [ 7.        ,  7.        ,  7.        ,  7.        ],
           [ 1.52942389,  0.37328854,  0.        ,  0.        ],
           [ 0.        ,  1.40398143,  0.324043  ,  0.        ]])



## 花式索引


```python
# 花式索引
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr
```




    array([[ 0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.],
           [ 2.,  2.,  2.,  2.],
           [ 3.,  3.,  3.,  3.],
           [ 4.,  4.,  4.,  4.],
           [ 5.,  5.,  5.,  5.],
           [ 6.,  6.,  6.,  6.],
           [ 7.,  7.,  7.,  7.]])




```python
arr[[4, 3, 0, 6]]
```




    array([[ 4.,  4.,  4.,  4.],
           [ 3.,  3.,  3.,  3.],
           [ 0.,  0.,  0.,  0.],
           [ 6.,  6.,  6.,  6.]])




```python
arr[[-3, -5, -7]]
```




    array([[ 5.,  5.,  5.,  5.],
           [ 3.,  3.,  3.,  3.],
           [ 1.,  1.,  1.,  1.]])




```python
arr = np.arange(32).reshape((8, 4))
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31]])




```python
arr[[1, 5, 7, 2], [0, 3, 1, 2]]
```




    array([ 4, 23, 29, 10])




```python
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]
```




    array([[ 4,  7,  5,  6],
           [20, 23, 21, 22],
           [28, 31, 29, 30],
           [ 8, 11,  9, 10]])




```python
arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]
```




    array([[ 4,  7,  5,  6],
           [20, 23, 21, 22],
           [28, 31, 29, 30],
           [ 8, 11,  9, 10]])



## 数组转置


```python
# 数组转置
arr = np.arange(15).reshape((3, 5))
arr
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])




```python
arr.T
```




    array([[ 0,  5, 10],
           [ 1,  6, 11],
           [ 2,  7, 12],
           [ 3,  8, 13],
           [ 4,  9, 14]])



## 改变数组的维度


```python
# 改变数组的维度
b = np.arange(24).reshape(2, 3, 4)

print(b)

print(b.ravel())

print(b.flatten())

b.shape = (6, 4)

print(b)

print(b.transpose())

b.resize((2, 12))

print(b)
```

    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
    
     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]]
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]
     [16 17 18 19]
     [20 21 22 23]]
    [[ 0  4  8 12 16 20]
     [ 1  5  9 13 17 21]
     [ 2  6 10 14 18 22]
     [ 3  7 11 15 19 23]]
    [[ 0  1  2  3  4  5  6  7  8  9 10 11]
     [12 13 14 15 16 17 18 19 20 21 22 23]]
    

## 组合数组


```python
# 组合数组
a = np.arange(9).reshape(3, 3)

print(a)

b = 2 * a

print(b)

print(np.hstack((a, b)))

print(np.concatenate((a, b), axis=1))

print(np.vstack((a, b)))

print(np.concatenate((a, b), axis=0))

print(np.dstack((a, b)))

oned = np.arange(2)

print(oned)

twice_oned = 2 * oned

print(twice_oned)

print(np.column_stack((oned, twice_oned)))

print(np.column_stack((a, b)))

print(np.column_stack((a, b)) == np.hstack((a, b)))

print(np.row_stack((oned, twice_oned)))

print(np.row_stack((a, b)))

print(np.row_stack((a, b)) == np.vstack((a, b)))
```

    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    [[ 0  2  4]
     [ 6  8 10]
     [12 14 16]]
    [[ 0  1  2  0  2  4]
     [ 3  4  5  6  8 10]
     [ 6  7  8 12 14 16]]
    [[ 0  1  2  0  2  4]
     [ 3  4  5  6  8 10]
     [ 6  7  8 12 14 16]]
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 0  2  4]
     [ 6  8 10]
     [12 14 16]]
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 0  2  4]
     [ 6  8 10]
     [12 14 16]]
    [[[ 0  0]
      [ 1  2]
      [ 2  4]]
    
     [[ 3  6]
      [ 4  8]
      [ 5 10]]
    
     [[ 6 12]
      [ 7 14]
      [ 8 16]]]
    [0 1]
    [0 2]
    [[0 0]
     [1 2]]
    [[ 0  1  2  0  2  4]
     [ 3  4  5  6  8 10]
     [ 6  7  8 12 14 16]]
    [[ True  True  True  True  True  True]
     [ True  True  True  True  True  True]
     [ True  True  True  True  True  True]]
    [[0 1]
     [0 2]]
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 0  2  4]
     [ 6  8 10]
     [12 14 16]]
    [[ True  True  True]
     [ True  True  True]
     [ True  True  True]
     [ True  True  True]
     [ True  True  True]
     [ True  True  True]]
    

## 数组的分割


```python
# 数组的分割
a = np.arange(6).reshape(2, 3)

print(a)

print(np.hsplit(a, 3))

print(np.split(a, 3, axis=1))

print(np.vsplit(a, 2))

print(np.split(a, 2, axis=0))

c = np.arange(27).reshape(3, 3, 3)

print(c)

print(np.dsplit(c, 3))
```

    [[0 1 2]
     [3 4 5]]
    [array([[0],
           [3]]), array([[1],
           [4]]), array([[2],
           [5]])]
    [array([[0],
           [3]]), array([[1],
           [4]]), array([[2],
           [5]])]
    [array([[0, 1, 2]]), array([[3, 4, 5]])]
    [array([[0, 1, 2]]), array([[3, 4, 5]])]
    [[[ 0  1  2]
      [ 3  4  5]
      [ 6  7  8]]
    
     [[ 9 10 11]
      [12 13 14]
      [15 16 17]]
    
     [[18 19 20]
      [21 22 23]
      [24 25 26]]]
    [array([[[ 0],
            [ 3],
            [ 6]],
    
           [[ 9],
            [12],
            [15]],
    
           [[18],
            [21],
            [24]]]), array([[[ 1],
            [ 4],
            [ 7]],
    
           [[10],
            [13],
            [16]],
    
           [[19],
            [22],
            [25]]]), array([[[ 2],
            [ 5],
            [ 8]],
    
           [[11],
            [14],
            [17]],
    
           [[20],
            [23],
            [26]]])]
    

## 数组的属性


```python
# 数组的属性
b = np.arange(24).reshape(2, 12)
b.ndim
b.size
b.itemsize
b.nbytes

b = np.array([1. + 1.j, 3. + 2.j])
b.real
b.imag

b = np.arange(4).reshape(2, 2)
b.flat
b.flat[2]
```




    2



## 数组的转换


```python
# 数组的转换
b = np.array([1. + 1.j, 3. + 2.j])
print(b)

print(b.tolist())

print(b.tostring())

print(np.fromstring(
    b'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x00@',
    dtype=complex))

print(np.fromstring('20:42:52', sep=':', dtype=int))

print(b)

print(b.astype(int))

print(b.astype('complex'))
```

    [ 1.+1.j  3.+2.j]
    [(1+1j), (3+2j)]
    b'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x00@'
    [ 1.+1.j  3.+2.j]
    [20 42 52]
    [ 1.+1.j  3.+2.j]
    [1 3]
    [ 1.+1.j  3.+2.j]
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:17: ComplexWarning: Casting complex values to real discards the imaginary part
    

参考资料：炼数成金Python数据分析课程
