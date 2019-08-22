
# 条件语句

## if...elif...else


```python
# 条件语句
'''
if 判断条件：
    执行语句……
else：
    执行语句……
'''

flag = False
name = 'python'
if name == 'python':  # 判断变量否为'python'
    flag = True  # 条件成立时设置标志为真
    print('welcome boss')  # 并输出欢迎信息
else:
    print(name)  # 条件不成立时输出变量名称
```

    welcome boss
    


```python
'''
if 判断条件1:
    执行语句1……
elif 判断条件2:
    执行语句2……
elif 判断条件3:
    执行语句3……
else:
    执行语句4……
'''

num = 2
if num == 3:  # 判断num的值
    print('boss')
elif num == 2:
    print('user')
elif num == 1:
    print('worker')
elif num < 0:  # 值小于零时输出
    print('error')
else:
    print('roadman')  # 条件均不成立时输出
```

    user
    


```python
num = 9
if num >= 0 and num <= 10:  # 判断值是否在0~10之间
    print('hello')

if 0 <= num <= 10:  # num >= 0 and num <= 10 简化写法
    print('hello')
```

    hello
    hello
    


```python
num = 10
if num < 0 or num > 10:  # 判断值是否在小于0或大于10
    print('hello')
else:
    print('undefine')
```

    undefine
    


```python
num = 8
# 判断值是否在0~5或者10~15之间
if (num >= 0 and num <= 5) or (num >= 10 and num <= 15):
    print('hello')
else:
    print('undefine')
```

    undefine
    


```python
var = 100
if var == 100: print("变量 var 的值为100")
print("Good bye!")
```

    变量 var 的值为100
    Good bye!
    

# 循环语句

## while语句


```python
# while语句
'''
while 判断条件：
    执行语句……
'''
count = 0
while count < 9:
    print('The count is:', count)
    count = count + 1

print("Good bye!")

```

    The count is: 0
    The count is: 1
    The count is: 2
    The count is: 3
    The count is: 4
    The count is: 5
    The count is: 6
    The count is: 7
    The count is: 8
    Good bye!
    

continue


```python
# continue 和 break 用法

i = 1
while i < 10:
    i += 1
    if i % 2 > 0:  # 非双数时跳过输出
        continue
    print(i)  # 输出双数2、4、6、8、10
```

    2
    4
    6
    8
    10
    

break


```python
i = 1
while 1:  # 循环条件为1必定成立
    print(i)  # 输出1~10
    i += 1
    if i > 10:  # 当i大于10时跳出循环
        break
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    

死循环


```python
# 死循环
# 按 ctrl+c 可以强制结束死循环
'''
var = 1
while var == 1 :  # 该条件永远为true，循环将无限执行下去
   num = raw_input("Enter a number  :")
   print "You entered: ", num

print "Good bye!"
'''
```




    '\nvar = 1\nwhile var == 1 :  # 该条件永远为true，循环将无限执行下去\n   num = raw_input("Enter a number  :")\n   print "You entered: ", num\n\nprint "Good bye!"\n'



while … else


```python
# while … else
count = 0
while count < 5:
    print(count, " is  less than 5")
    count = count + 1
else:
    print(count, " is not less than 5")

```

    0  is  less than 5
    1  is  less than 5
    2  is  less than 5
    3  is  less than 5
    4  is  less than 5
    5  is not less than 5
    


```python
# 简单语句组
flag = 1
while flag: print('Given flag is really true!');flag = 0;
print("Good bye!")
```

    Given flag is really true!
    Good bye!
    

## for语句


```python
# for语句
'''
for iterating_var in sequence:
   statements(s)
'''
for letter in 'Python':  # 第一个实例
    print('当前字母 :', letter)
```

    当前字母 : P
    当前字母 : y
    当前字母 : t
    当前字母 : h
    当前字母 : o
    当前字母 : n
    


```python
fruits = ['banana', 'apple', 'mango']
for fruit in fruits:  # 第二个实例
    print('当前水果 :', fruit)

print("Good bye!")
```

    当前水果 : banana
    当前水果 : apple
    当前水果 : mango
    Good bye!
    


```python
# 序列索引迭代
fruits = ['banana', 'apple', 'mango']
for index in range(len(fruits)):
    print('当前水果 :', fruits[index])

print("Good bye!")
```

    当前水果 : banana
    当前水果 : apple
    当前水果 : mango
    Good bye!
    

for...else


```python
# for...else
for num in range(10, 20):  # 迭代 10 到 20 之间的数字
    for i in range(2, num):  # 根据因子迭代
        if num % i == 0:  # 确定第一个因子
            j = num / i  # 计算第二个因子
            print('%d 等于 %d * %d' % (num, i, j))
            break  # 跳出当前循环
    else:  # 循环的 else 部分
        print(num, '是一个质数')
```

    10 等于 2 * 5
    11 是一个质数
    12 等于 2 * 6
    13 是一个质数
    14 等于 2 * 7
    15 等于 3 * 5
    16 等于 2 * 8
    17 是一个质数
    18 等于 2 * 9
    19 是一个质数
    

## 嵌套循环


```python
# 嵌套循环
i = 2
while (i < 100):
    j = 2
    while (j <= (i / j)):
        if not (i % j): break
        j = j + 1
    if (j > i / j): print(i, " 是素数")
    i = i + 1

print("Good bye!")
```

    2  是素数
    3  是素数
    5  是素数
    7  是素数
    11  是素数
    13  是素数
    17  是素数
    19  是素数
    23  是素数
    29  是素数
    31  是素数
    37  是素数
    41  是素数
    43  是素数
    47  是素数
    53  是素数
    59  是素数
    61  是素数
    67  是素数
    71  是素数
    73  是素数
    79  是素数
    83  是素数
    89  是素数
    97  是素数
    Good bye!
    


```python
# break语句
for letter in 'Python':  # First Example
    if letter == 'h':
        break
    print('Current Letter :', letter)
```

    Current Letter : P
    Current Letter : y
    Current Letter : t
    


```python
var = 10  # Second Example
while var > 0:
    print('Current variable value :', var)
    var = var - 1
    if var == 5:
        break

print("Good bye!")
```

    Current variable value : 10
    Current variable value : 9
    Current variable value : 8
    Current variable value : 7
    Current variable value : 6
    Good bye!
    


```python
# continue语句
for letter in 'Python':  # 第一个实例
    if letter == 'h':
        continue
    print('当前字母 :', letter)
```

    当前字母 : P
    当前字母 : y
    当前字母 : t
    当前字母 : o
    当前字母 : n
    


```python
var = 10  # 第二个实例
while var > 0:
    var = var - 1
    if var == 5:
        continue
    print('当前变量值 :', var)
print("Good bye!")
```

    当前变量值 : 9
    当前变量值 : 8
    当前变量值 : 7
    当前变量值 : 6
    当前变量值 : 4
    当前变量值 : 3
    当前变量值 : 2
    当前变量值 : 1
    当前变量值 : 0
    Good bye!
    

pass语句


```python
# pass语句
# 输出 Python 的每个字母
for letter in 'Python':
    if letter == 'h':
        pass
        print('这是 pass 块')
    print('当前字母 :', letter)

print("Good bye!")
```

    当前字母 : P
    当前字母 : y
    当前字母 : t
    这是 pass 块
    当前字母 : h
    当前字母 : o
    当前字母 : n
    Good bye!
    

# 其他

## 格式字符串


```python
# 格式字符串
print("My name is %s and weight is %d kg!" % ('Zara', 21))
```

    My name is Zara and weight is 21 kg!
    

## 时间与日期


```python
# 时间与日期
import time  # This is required to include time module.

ticks = time.time()
print("Number of ticks since 12:00am, January 1, 1970:", ticks)

localtime = time.localtime(time.time())
print("Local current time :", localtime)

localtime = time.asctime(time.localtime(time.time()))
print("Local current time :", localtime)
```

    Number of ticks since 12:00am, January 1, 1970: 1564649998.4399576
    Local current time : time.struct_time(tm_year=2019, tm_mon=8, tm_mday=1, tm_hour=16, tm_min=59, tm_sec=58, tm_wday=3, tm_yday=213, tm_isdst=0)
    Local current time : Thu Aug  1 16:59:58 2019
    


```python
import calendar

cal = calendar.month(2008, 1)
print("Here is the calendar:")
print(cal)
```

    Here is the calendar:
        January 2008
    Mo Tu We Th Fr Sa Su
        1  2  3  4  5  6
     7  8  9 10 11 12 13
    14 15 16 17 18 19 20
    21 22 23 24 25 26 27
    28 29 30 31
    
    

## 自定义函数


```python
# 自定义函数
'''
def functionname( parameters ):
   "函数_文档字符串"
   function_suite
   return [expression]		
'''


def printme(str):
    """打印传入的字符串到标准显示设备上"""
    print(str)
    return

```


```python
# 函数调用
printme("我要调用用户自定义函数!");
printme("再次调用同一函数");
```

    我要调用用户自定义函数!
    再次调用同一函数
    


```python
# 可写函数说明
def changeme(mylist):
    """修改传入的列表"""
    mylist.append([1, 2, 3, 4]);
    print("函数内取值: ", mylist)
    return
```


```python
# 调用changeme函数
mylist = [10, 20, 30];
changeme(mylist);
print("函数外取值: ", mylist)
```

    函数内取值:  [10, 20, 30, [1, 2, 3, 4]]
    函数外取值:  [10, 20, 30, [1, 2, 3, 4]]
    


```python
# 参数
def printme(str):
    """打印任何传入的字符串"""
    print(str)
    return

# 调用printme函数
printme(str="My string")
```

    My string
    


```python
def printinfo(name, age):
    """打印任何传入的字符串"""
    print("Name: ", name)
    print("Age ", age)
    return

# 调用printinfo函数
printinfo(age=50, name="miki")
```

    Name:  miki
    Age  50
    


```python
def printinfo(name, age=35):
    """打印任何传入的字符串"""
    print("Name: ", name)
    print("Age ", age)
    return


# 调用printinfo函数
printinfo(age=50, name="miki");
printinfo(name="miki");
```

    Name:  miki
    Age  50
    Name:  miki
    Age  35
    


```python
# 不定长参数
'''
def functionname([formal_args,] *var_args_tuple ):
   "函数_文档字符串"
   function_suite
   return [expression]
'''


def printinfo(arg1, *vartuple):
    """打印任何传入的字符串"""
    print("输出: ")
    print(arg1)
    for var in vartuple:
        print(var)
    return


# 调用printinfo 函数
printinfo(10)
printinfo(70, 60, 50)
```

    输出: 
    10
    输出: 
    70
    60
    50
    

## 匿名函数


```python
# 匿名函数
'''
lambda [arg1 [,arg2,.....argn]]:expression
'''

sum = lambda arg1, arg2: arg1 + arg2
# 调用sum函数
print("相加后的值为 : ", sum(10, 20))
print("相加后的值为 : ", sum(20, 20))
```

    相加后的值为 :  30
    相加后的值为 :  40
    

## return语句


```python
# return语句
def sum(arg1, arg2):
    # 返回2个参数的和."
    total = arg1 + arg2
    print("函数内 : ", total)
    return total
```


```python
# 调用sum函数
total = sum(10, 20)
print("函数外 : ", total)
```

    函数内 :  30
    函数外 :  30
    


```python
# 变量的作用范围
total = 0  # 这是一个全局变量
```


```python
# 可写函数说明
def sum(arg1, arg2):
    # 返回2个参数的和.
    total = arg1 + arg2  # total在这里是局部变量.
    print("函数内是局部变量 : ", total)
    return total


# 调用sum函数
sum(10, 20)
print("函数外是全局变量 : ", total)
```

    函数内是局部变量 :  30
    函数外是全局变量 :  0
    

## 键盘输入


```python
# 键盘输入
str = input("Please enter:")
print("你输入的内容是: ", str)
```

## 打开与关闭文件


```python
# 打开与关闭文件
# 打开一个文件
fo = open("data/foo.txt", "wb")
print("文件名: ", fo.name)
print("是否已关闭 : ", fo.closed)
print("访问模式 : ", fo.mode)
# 关闭打开的文件
fo.close()
```

    文件名:  data/foo.txt
    是否已关闭 :  False
    访问模式 :  wb
    


```python
# 打开一个文件
fo = open("data/foo.txt", "wb")
fo.write("www.runoob.com!\nVery good site!\n")

# 关闭打开的文件
fo.close()
```


```python
# 打开一个文件
fo = open("data/foo.txt", "r+")
str = fo.read(10)
print("读取的字符串是 : ", str)
# 关闭打开的文件
fo.close()
```

    读取的字符串是 :  
    


```python
# 打开一个文件
fo = open("data/foo.txt", "r+")
str = fo.read(10)
print("读取的字符串是 : ", str)

# 查找当前位置
position = fo.tell()
print("当前文件位置 : ", position)

# 把指针再次重新定位到文件开头
position = fo.seek(0, 0)
str = fo.read(10)
print("重新读取字符串 : ", str)
# 关闭打开的文件
fo.close()
```

    读取的字符串是 :  
    当前文件位置 :  0
    重新读取字符串 :  
    


```python
import os

# 重命名文件test1.txt到test2.txt。
os.rename("data/test1.txt", "data/test2.txt")
```


```python
# 删除一个已经存在的文件test2.txt
os.remove("data/test2.txt")
```

## 异常处理


```python
# 异常处理
try:
    fh = open("data/testfile", "w")
    fh.write("This is my test file for exception handling!!")
except IOError:
    print("Error: can\'t find file or read data")
else:
    print("Written content in the file successfully")
    fh.close()
```

    Written content in the file successfully
    


```python
try:
    fh = open("data/testfile", "r")
    fh.write("This is my test file for exception handling!!")
except IOError:
    print("Error: can\'t find file or read data")
else:
    print("Written content in the file successfully")
```

    Error: can't find file or read data
    


```python
try:
    fh = open("data/testfile", "w")
    fh.write("This is my test file for exception handling!!")
finally:
    print("Error: can\'t find file or read data")
```

    Error: can't find file or read data
    


```python
try:
    fh = open("data/testfile", "w")
    try:
        fh.write("This is my test file for exception handling!!")
    finally:
        print("Going to close the file")
        fh.close()
except IOError:
    print("Error: can\'t find file or read data")
```

    Going to close the file
    


```python
def temp_convert(var):
    try:
        return int(var)
    except ValueError as Argument:
        print("The argument does not contain numbers\n", Argument)


# Call above function here.
temp_convert("xyz")
```

    The argument does not contain numbers
     invalid literal for int() with base 10: 'xyz'
    


```python
# 异常触发
def functionName(level):
    if level < 1:
        raise Exception("Invalid level!")
        # The code below to this would not be executed
        # if we raise the exception


try:
    # Business Logic here...
    functionName(0)
except Exception as e:
    # Exception handling here...
    print(e)
else:
    # Rest of the code here...
    print("No Exception")
```

    Invalid level!
    

## 自定义异常


```python
# 自定义异常
class NetworkError(RuntimeError):
    def __init__(self, arg):
        self.args = arg


try:
    raise NetworkError("Bad hostname")
except NetworkError as e:
    print(e.args)
```

    ('B', 'a', 'd', ' ', 'h', 'o', 's', 't', 'n', 'a', 'm', 'e')
    

参考资料：炼数成金Python数据分析课程
