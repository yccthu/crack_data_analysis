# Python数据分析萌新避坑指南Task04：数据可视化

大家好，又到了新的萌新日常来找茬的时刻了！

在开始之前默念三遍：

莫生气！

莫生气！

莫生气！

那就开始吧！

前面几个task我们进行了简单的描述性数理统计分析，但是若要看到具体的趋势和走向，或者更直观的了解数据集的特点，我们就需进行数据可视化，把一个个数字映射到图上。

# 1. 准备工作

一如既往导入我们赖以生存的numpy和pandas！今天介绍一个新成员，叫matplotlib，他带来了很多我们整不明白，但最终都会用的函数！


```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt #导入2D画图库
from numpy.random import randn #因为第二部分的练习，我们没有数据集，所以我们会用随机漫步的这个包包，来让数据自由游走
```

有了包包，我们还需要魔法，下面这个inline小魔法是画图必备的插件，有了它才能显示你的图，你不会只想画不想看的呢。


```python
%matplotlib inline
```

导入上期乱七八糟合并完了的数据集，我们就可以开始开始....**学习基础**了

【满脸写着高兴.jpg】


```python
titan = pd.read_csv('result.csv')
```


```python
titan.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



# 2 如何让人一眼看懂你的数据？

## 2.1 任务一：《Python for Data Analysis》第九章

这部分主要是跟着《利用python进行数据分析》进行一些基本的操作，我也会给大家进行萌新尽了全力的秃头小拓展和小优化。


```python
data = np.arange(10) #生成1-10的数列
data
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
plt.plot(data) #先随意的画一个折线图
```




    [<matplotlib.lines.Line2D at 0x2bd8285a0f0>]




![png](output_14_1.png)


### **Matplotlib API绘图**

### 2.1.1 Figure和Subplot

在实际应用，尤其是当你进行科研作图的时候，一幅图往往不足以呈现你的研究成果，更好的展示方式是一组图上进行对比，这就是我们俗称的图集，图集是由子图组合来实现的.

Matplotlib将图称为figure，一个figure相当于一张白色的大画布，我们要在这张画布开展制图的工作。命令`figure()`会让我们看到目前画布的大小，如下，我们创建了一张大小为423*288像素的画布。


```python
fig = plt.figure()
```


    <Figure size 432x288 with 0 Axes>


画布是无法直接画图的，我们需要把要画的内容一块一块的加入到画布里，用add_subplot()的命令，把子图加入空画布中。


```python
ax1 = fig.add_subplot(2,2,1)
```

2,2,1 表示的是每行两个图，每列两个图，然后我们从第一个图开始画画。同样的，我们也需要在第二个图，第三个图和第四图画画。

【注】
- 实际科研作图中，每次绘图都会覆盖原画面，因此，我们需要在一个框框里完整图像的绘制，以此来保证图像的准确和完整。
- 由于我们没有实验用的数据集，这里的数据集是用随机漫步random walk生成的，随机漫步的包包是numpy.random，我们缩写为randn，这样的数据集在进行科学实验的时候尤其有用。但作为例子的时候，每次生成的新图都有可能和本文萌新画出来的不一样哦！

【思路一】


```python
fig = plt.figure() #先新建一张画布，记得一定要赋值！！！一定要赋值！！！一定要赋值！！！

#根据每块布的坐标来画
ax1 = fig.add_subplot(2,2,1) 
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

#画图1
ax1.plot(np.random.randn(50).cumsum(),'k--')
#画图2
ax2.hist(np.random.randn(100),bins = 20, color = 'k', alpha = 0.3)
#画图3
ax3.scatter(np.arange(30), np.arange(30)+3*np.random.randn(30))

#给我瞅瞅
plt.show()
```


![png](output_24_0.png)


【思路2】


这里axes相当于一个二维数组，可以对图坐标进行索引


```python
x = np.arange(0,100)

# 划分子图

fig, axes = plt.subplots(2,2) #做一个2X2的四个小图图的画布(✿◡‿◡)

#把这四个小图图按照坐标分别放到这张画布里
ax1 =  axes[0,0]
ax2 =  axes[0,1]
ax3 =  axes[1,0]
ax4 =  axes[1,1]

# 一个一个来画
# 画图1
ax1.plot(x,x)
# 画图2
ax2.plot(x,-x)
# 画图3
ax3.plot(x,x**2)
# 画图4
ax4.plot(x, np.log(x))

#魔法的展示时刻

plt.show()
```

    d:\anaconda3\envs\python3.6\lib\site-packages\ipykernel_launcher.py:21: RuntimeWarning: divide by zero encountered in log
    


![png](output_27_1.png)


### 2.1.2 调整subplot间距

调整子图的间距用的是以下参数：

```
subplot_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = None)
```

为了方便多图比较，我们还要进行以下设置

- sharex: 所有的子图使用相同的x轴刻度
- sharey: 所有的子图使用相同的y轴可图



```python
fig,axes = plt.subplots(2,2, sharex = True, sharey= True)

for i in range(2):
    for j in range(2):
        axes[i,j].hist(np.random.randn(500), bins=50, color= 'k', alpha = 0.5)

plt.subplots_adjust(wspace=0, hspace=0)
```


![png](output_32_0.png)


### 2.1.3 颜色标记和线形


```python
from numpy.random import randn #导入numpy.random里面随机漫步用的包包！
```


```python
plt.plot(randn(30).cumsum(), 'ko--')
```




    [<matplotlib.lines.Line2D at 0x2bd95909b00>]




![png](output_35_1.png)



```python
plt.plot(randn(30).cumsum(), color = 'k', linestyle = 'dashed', marker = 'o')
```




    [<matplotlib.lines.Line2D at 0x2bd95ebf630>]




![png](output_36_1.png)



```python
data1 = np.random.randn(30).cumsum()
plt.plot(data1, 'k--', label = 'Default')
```




    [<matplotlib.lines.Line2D at 0x2bd95c78860>]




![png](output_37_1.png)


用`drawstyle = ' '`参数来设置类型


```python
data2 = np.random.randn(30).cumsum()
plt.plot(data2, 'k-', drawstyle = 'steps-post', label = 'steps-post')
```




    [<matplotlib.lines.Line2D at 0x2bd95e5d438>]




![png](output_39_1.png)


把两个随机漫步的数据合在一张图上画，`plt.legent()`会帮助你创建图例，这是调用图例的唯一办法，**没有捷径**！


```python
data1 = np.random.randn(30).cumsum()
plt.plot(data1, 'k--', label = 'Default')
plt.plot(data1, 'k-', drawstyle = 'steps-post', label = 'steps-post')

#图例放在最合适的位置
plt.legend(loc = 'best')
```




    <matplotlib.legend.Legend at 0x2bd95aec3c8>




![png](output_41_1.png)


### 2.1.4 刻度，标签和图例

我们可以用`set_xticks`来设置划分的数值，用`set_xticklabels`来设置标签， `fontsize`设置标签大小，`rotation`设置倾斜程度。

【思路一】


```python
#画图部分
fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum())

#设置图标题
ax.set_title('Happy Random Walk')

# 设置x轴的特性
xticks = ax.set_xticks([0,250,500,750,1000])
xt_labels = ax.set_xticklabels(['one','two','three','four','five'], rotation = 30, fontsize = 'small')

#设置y轴标签

ax.set_xlabel('Stages')

#设置y轴的特性
yticks = ax.set_yticks([-20,0,20,40])
yt_labels = ax.set_yticklabels(['-20','0','20','40'], rotation = 30, fontsize = 'small')

#设置y轴标签
ax.set_ylabel('Values')

```




    Text(0, 0.5, 'Values')




![png](output_45_1.png)


【思路二】

一次性设置各个轴标签和标题的方法， 用`ax.set`


```python
#画图部分
fig2 = plt.figure()
ax = fig2.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum())

#设置各个轴标签 ，注意这是一个类，每一个标签之间，一定要打上逗号，不然就会在冒号处一直报错！
labels = {
    'title': 'Miserable Random Walk',
    'xlabel': 'Stages',
    'ylabel': 'Values'
}

ax.set(**labels)

# 设置x轴的特性
xticks = ax.set_xticks([0,250,500,750,1000])
xt_labels = ax.set_xticklabels(['one','two','three','four','five'], rotation = 30, fontsize = 'small')

#设置y轴的特性
yticks = ax.set_yticks([-20,0,20,40])
yt_labels = ax.set_yticklabels(['-20','0','20','40'], rotation = 30, fontsize = 'small')

```


![png](output_48_0.png)


### 2.1.5 添加图例

之前简单的介绍了以下，`ax.legend()`和`plt.legend()`都可以用来添加图例,添加子图的时候，顺手传一个legend参数.

图例得位置是用一个叫`loc`的参数来设定的，一般情况下用`best`，`best`会根据数据图形变化，找到相应的空地，放置图例。


```python
fig3 = plt.figure();ax = fig3.add_subplot(1,1,1) 
# 注意用;隔开的新的书写方式，虽可以省去一点点的书写空间，但不是规范的写法哦！《数据分析》P258

ax.plot(randn(1000).cumsum(), 'k', label = 'one')
ax.plot(randn(1000).cumsum(), 'k--', label = 'two')
ax.plot(randn(1000).cumsum(), 'k.', label = 'three')

ax.legend(loc = 'best')
```




    <matplotlib.legend.Legend at 0x2bd9606fd30>




![png](output_51_1.png)


### 2.1.6 注释及subplot上绘图（科研刚需——待更新）

我们需要把显著的实验结果在图上凸出显示，让读者或者审稿人更好的看到我们观测到的结果。利用注释和子图上的绘图能更好的实现我们的目标。

注释的方式有三种：

- `text()` 文本添加
- `arrow()` 箭头添加
- `annotate()`注释添加

添加函数的参数如下：

```
ax.text(x,y, 'Hello, World!', family = 'monospace', fontsize =10 )
```

我们在这里用一下泰坦尼克号的数据，我们熟悉的titan中的生存人数按照票价来画图。


```python
# 计算不同票价中生存与死亡人数 1表示生存，0表示死亡
fare_sur = titan.groupby(['Fare'])['Survived'].value_counts().sort_values(ascending=False)
fare_sur
```




    Fare     Survived
    8.0500   0           38
    7.8958   0           37
    13.0000  0           26
    7.7500   0           22
    26.0000  0           16
                         ..
    20.2500  1            1
             0            1
    18.7875  1            1
             0            1
    15.0500  0            1
    Name: Survived, Length: 330, dtype: int64




```python
titan['Fare'].mean()
```




    32.204207968574636




```python
#绘制这些图

#建设画布一块
fig = plt.figure(figsize=(20, 18))

#画格子
fare_sur.plot(grid=True)

#做一个注释专用的字典，类

annotation = {
    titan['Fare'].mean(): 'Average Fare'
    titan['Fare'].max(): 'Max Fare'
    titan['Fare'].min(): 'Minimun Fare'
}


#然后我们要找到y得位置。。@#￥%……&*（ 这部分太难了，晚点再弄。。。
ax.annotate()

#显示图例和图！
plt.legend()
plt.show()
```


      File "<ipython-input-137-17606c5fc49e>", line 13
        titan['Fare'].max(): 'Max Fare'
            ^
    SyntaxError: invalid syntax
    


### **使用Pandas和Seaborn绘图**

书上对这两个库画图做了很多的解释，对于一个科研汪，我只想说：Seaborn真香！！！

### 2.1.7 线型图

折线图的命令`plt.plot()`，前面已经反复尝试，在这里就不讲了，值得注意的是一些每天都要用到的参数，比如`grid`网格，默认是打开的，还有y轴和x轴界限，ylim和xlin函数。

### 2.1.8 柱状图

命令 `plot.bar()`纵向柱状图和`plot.barh()`横向柱状图


```python
fig, axes = plt.subplots(2,1)

data5 = pd.Series(np.random.rand(16), index = list('abcdefghijklmnop')) #index设置一个每个柱子的标签

data5.plot.bar(ax = axes[0], color = 'k', alpha = 0.7) # axes[0]是图的位置， color设置了黑色，alpha是透明度为0.7
data5.plot.barh(ax = axes[1], color = 'k', alpha = 0.7)
```




    <AxesSubplot:>




![png](output_66_1.png)


实际分析的数据，总是由有更多的维度，例如下面的数据，将每一个总数的趋势和每一个单独分类的占比都有所表示。


```python
df = pd.DataFrame(np.random.rand(6,4),index = ['one','two','three','four','five','six'], columns = 
                  pd.Index(['A','B','C','D'],name = 'Genus'))
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Genus</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>0.401173</td>
      <td>0.149869</td>
      <td>0.196865</td>
      <td>0.888810</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.920717</td>
      <td>0.987153</td>
      <td>0.396678</td>
      <td>0.744742</td>
    </tr>
    <tr>
      <th>three</th>
      <td>0.850152</td>
      <td>0.217117</td>
      <td>0.931724</td>
      <td>0.888079</td>
    </tr>
    <tr>
      <th>four</th>
      <td>0.290783</td>
      <td>0.295332</td>
      <td>0.063040</td>
      <td>0.800424</td>
    </tr>
    <tr>
      <th>five</th>
      <td>0.977484</td>
      <td>0.474014</td>
      <td>0.845450</td>
      <td>0.481466</td>
    </tr>
    <tr>
      <th>six</th>
      <td>0.318926</td>
      <td>0.874089</td>
      <td>0.688984</td>
      <td>0.647529</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(2,1)
df.plot.barh(ax = axes[0], stacked=True, alpha = 0.5) 
df.plot.bar(ax = axes[1], stacked=True, alpha = 0.5)
```




    <AxesSubplot:>




![png](output_69_1.png)


### 2.1.9 直方图和密度图

直方图在面板数据的实证分析中常常是一项研究所需用到的第一幅图，查看数据的离散分布形式，以初步判断什么样的模型能更好拟合。

我们在下面泰坦尼克号数据集的分析中，会用到这部分的图。

## 2.2 任务二：可视化展示泰坦尼克号数据集中男女生存人数分布情况


```python
fig, axes = plt.subplots(2,1)

sex = titan.groupby('Sex')['Survived'].sum()
sex.plot.bar(ax = axes[0])
sex.plot.barh(ax = axes[1])

plt.subplots_adjust(hspace =1)

plt.title('survived_count')
plt.show()
```


![png](output_74_0.png)


【其他】如果想从数据上直观一点进行对比，我们也可以用value_counts()函数。


```python
# 1表示生存，0表示死亡

sex = titan.groupby('Sex')['Survived'].value_counts()
sex
```




    Sex     Survived
    female  1           233
            0            81
    male    0           468
            1           109
    Name: Survived, dtype: int64



【思考】计算出泰坦尼克号数据集中男女中死亡人数，并可视化展示？如何和男女生存人数可视化柱状图结合到一起？看到你的数据可视化，说说你的第一感受（比如：你一眼看出男生存活人数更多，那么性别可能会影响存活率）。

【答案】女性的生存人数几乎是男性生存人数的两倍。

从历史上我们知道，泰坦尼克号的船长是一个极具怜悯的角色，有可能是因为其号召男性对女性保护，让女性先上逃生艇，从而救助了更多的女性。当然，这样的定性分析从数据上很难获得。

## 2.3 任务三：可视化展示泰坦尼克号数据集中男女中生存人与死亡人数的比例图（用柱状图试试）。


```python
fig, axes = plt.subplots(1,2)

titan.groupby(['Sex','Survived'])['Survived'].count().unstack().plot(ax = axes[0], kind='bar',stacked='True')
titan.groupby(['Sex','Survived'])['Survived'].count().unstack().plot(ax = axes[1], kind='barh',stacked='True')

plt.title('survived_count')
plt.ylabel('count')

plt.subplots_adjust(wspace =0.4)

```


![png](output_80_0.png)


## 2.4 任务四：可视化展示泰坦尼克号数据集中不同票价的人生存和死亡人数分布情况。（用折线图试试）（横轴是不同票价，纵轴是存活人数）


```python
fare_sur = titan.groupby(['Fare'])['Survived'].value_counts().sort_values(ascending=False)
fare_sur
```




    Fare     Survived
    8.0500   0           38
    7.8958   0           37
    13.0000  0           26
    7.7500   0           22
    26.0000  0           16
                         ..
    20.2500  1            1
             0            1
    18.7875  1            1
             0            1
    15.0500  0            1
    Name: Survived, Length: 330, dtype: int64




```python
# 排序后绘折线图
fig = plt.figure(figsize=(10, 8))
fare_sur.plot(grid=True)
plt.legend()
plt.show()
```

    d:\anaconda3\envs\python3.6\lib\site-packages\pandas\plotting\_matplotlib\core.py:1192: UserWarning: FixedFormatter should only be used together with FixedLocator
      ax.set_xticklabels(xticklabels)
    


![png](output_83_1.png)



```python
# 排序前绘折线图
fare_sur1 = titan.groupby(['Fare'])['Survived'].value_counts()
fare_sur1
```




    Fare      Survived
    0.0000    0           14
              1            1
    4.0125    0            1
    5.0000    0            1
    6.2375    0            1
                          ..
    247.5208  1            1
    262.3750  1            2
    263.0000  0            2
              1            2
    512.3292  1            3
    Name: Survived, Length: 330, dtype: int64




```python
fig = plt.figure(figsize=(10, 8))
fare_sur1.plot(grid=True)
plt.legend()
plt.show()
```

    d:\anaconda3\envs\python3.6\lib\site-packages\pandas\plotting\_matplotlib\core.py:1192: UserWarning: FixedFormatter should only be used together with FixedLocator
      ax.set_xticklabels(xticklabels)
    


![png](output_85_1.png)


## 2.5 任务五：可视化展示泰坦尼克号数据集中不同仓位等级的人生存和死亡人员的分布情况。（用柱状图试试）


```python
# 1表示生存，0表示死亡
pclass_sur = titan.groupby(['Pclass'])['Survived'].value_counts()
pclass_sur
```




    Pclass  Survived
    1       1           136
            0            80
    2       0            97
            1            87
    3       0           372
            1           119
    Name: Survived, dtype: int64




```python
import seaborn as sns
sns.countplot(x="Pclass", hue="Survived", data=titan)
```




    <AxesSubplot:xlabel='Pclass', ylabel='count'>




![png](output_88_1.png)


【思考】看到这个前面几个数据可视化，说说你的第一感受和你的总结

【答案】男性比女性死亡人数更多，3号仓位死亡人数非常高，而1号仓位的死亡人数相对较少。

## 2.6 任务六：可视化展示泰坦尼克号数据集中不同年龄的人生存与死亡人数分布情况。(不限表达方式)


```python
facet = sns.FacetGrid(titan, hue="Survived",aspect=3)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, titan['Age'].max()))
facet.add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x2bdffb8d240>




![png](output_92_1.png)


## 2.7 任务七：可视化展示泰坦尼克号数据集中不同仓位等级的人年龄分布情况。（用折线图试试）


```python
titan.Age[titan.Pclass == 1].plot(kind='kde')
titan.Age[titan.Pclass == 2].plot(kind='kde')
titan.Age[titan.Pclass == 3].plot(kind='kde')
plt.xlabel("age")
plt.legend((1,2,3),loc="best")
```




    <matplotlib.legend.Legend at 0x2bd827915c0>




![png](output_94_1.png)


【思考】上面所有可视化的例子做一个总体的分析，你看看你能不能有自己发现

【答案】从上面的分析我们可以发现，从死亡总人数来看，死亡的人数超过存活的人数，大多数人并没能逃出泰坦尼克号。男性比女性死亡人数多，20岁-30岁是生死挣扎最为频繁的年龄段，而年轻的生命死亡率略高于年长的生命。三号仓位最为年轻，二号仓其次，一号仓最为年长。然而，我们缺惊奇的发现，年长的仓位存活率更高，年轻的仓位死亡率远远高于其他客舱。从票价来看，大多数死亡的人来自于低票价的仓位，票价越高的仓位死亡人数越少，这也有可能是高票价仓位的售票数量本来就少导致的统计偏差。


```python

```
