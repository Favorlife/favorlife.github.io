---
layout: post
read_time: true
show_date: true
title:  Perceptron
date:   2022-10-16 22:29:20 +0800
description: Perceptron.
img: posts/20210228/MLLibrary.jpg 
tags: [machine learning, coding]
author: White Cool
github: favorlife/
---

# 感知机

- 二类分类的线性分类模型，属判别模型
- 旨在求出将训练数据进行==线性==划分的分离超平面
- 导入基于==误分类==的损失函数，利用==梯度下降法==对损失函数进行极小化
- 分为==原始形式==和==对偶形式==
- 是神经网络与支持向量机的基础



## 1. 感知机模型

**定义(感知机)**：假设输入空间(特征空间)是$X \subseteq R^{n}$, 输出空间是$Y = \{+1, -1\}$.其中，$x \in X$ 表示实例的特征向量， 输出$y \in Y$表示实例的类别。所表示的函数(1.1)如下：
$$
f(x) = sign(w\cdot x + b)
$$
$w$ 和 $b$ 为感知机模型参数，$w \in R^{n}$ 叫作权值(weight) 或权值向量(weight vector), $ b \in R$叫作偏置(bias), $w \cdot x$表示$w$ 和 $x$的内积。 $sign$是符号函数，即
$$
sign(x) = 
\begin{cases} 
		+1, & x\geq0\\ 
		-1, & x < 0 
\end{cases}
$$


## 2. 感知机学习策略

### 2.1 数据集的线性可分性

**定义(数据集的线性可分性)**: 给定一个数据集
$$
T = \{(x_1,y_1),(x_2,y_2),(x_3,y_3),\dots,(x_N,y_N)\}
$$
其中，$x_i \in R^n,y_i \in Y = \{+1, -1\}, i=1,2,\dots,N,$ 如果存在某个超平面 $S$ 能够将数据集的正实例点和负实例点完全正确划分到超平面的两侧，即对所有 $ y_i = +1$的实例$i$, 有$w\cdot x_i + b > 0$, 对所有$ y_i = -1$ 的实例$i$, 有$w\cdot x_i +b < 0$, 则称数据集 $T$ 为线性可分数据集，否则，称数据集 $T$ 为线性不可分。

### 2.2 感知机学习策略

- 明确学习目标: 求得一个能够将训练集==正实例点==和==负实例点==完全正确分开的分离超平面。
- 定义==(经验)==损失函数并将损失函数极小化

==**步骤：**==

1. 写出输入空间 $R^n$ 中的任一点 $x_0$ 到超平面 $S$ 的距离：
   $$
   \frac{1}{||w||}|w\cdot x_0 + b|
   $$
   这里，$||w||$ 是 $w$ 的 $L_2$ 范数==（不太理解范数和$L_2$是什么、、、就暂时当作距离公式得了）==

2. 对==误分类==的数据 $(x_i, y_i)$ 来说（注意是误分类，和正分类相反，故会得到如下结果），

$$
-y_i(w \cdot x_i + b )>0
$$

​	成立。由于当 $w\cdot x_i +b > 0$ 时，$y_i = -1$ (误分类导致符号相反， 正分类的话这里的结果是 1). 而$w\cdot x_i +b < 0$ 时，$y_i = +1$.故，误分类 $x_i$ 到 超平面 $S$ 的总距离:
$$
-\frac{1}{||w||}\sum_{x_i \in M} y_i(w\cdot x_i + b)
$$

3. 若不考虑 $\frac{1}{||w||}$, 就得到感知机的损失函数。故感知机$ sign(w\cdot x+b)$ 学习的损失函数(2.1)定义为：
   $$
   L(w, b) =-\sum_{x_i \in M} y_i(w\cdot x_i + b)
   $$
   其中，$M$ 为误分类点的集合。损失函数 $L(w,b)$  是 $w, b$ 的连续可导函数。



## 3. 感知机学习算法

- 感知机学习问题转化为求解损失函数(2.1)的最优化问题
- 最优化的方法是随机梯度下降法

==**问题描述:**==
	给定一个数据集
$$
T = \{(x_1,y_1),(x_2,y_2),(x_3,y_3),\dots,(x_N,y_N)\}
$$
其中，$x_i \in R^n,y_i \in Y = \{+1, -1\}, i=1,2,\dots,N,$ 求参数 $w , b$ , 使其为以下损失函数(3.1)极小化问题的解:
$$
\min_{w, b} L(w, b) =-\sum_{x_i \in M} y_i(w\cdot x_i + b)
$$
其中 $M$ 为误分类点的集合。



### 3.1 感知机学习算法的原始形式

 ==**基本思想**==(3.1):

1. 任意选取一个超平面 $w_0, b_0$ 

2. 用梯度下降法不断地极小化目标函数(3.1)

3. 极小化过程中不是一次使 $M$ 中的所有误分类点的梯度下降， 而是一次随机选取一个误分类点使其梯度下降。

4. 假如误分类点集合 $M$ 是固定的， 那么损失函数 $L(w, b)$ 的梯度由
   $$
   \bigtriangledown_{w} L(w,b) = - \sum_{x_i \in M}y_ix_i \\
   \bigtriangledown_{b} L(w,b) = - \sum_{x_i \in M}y_i
   $$
   给出.

5. 随机选取一个误分类点$(x_i, y_i)$ , 对 $w, b$ 进行更新:
   $$
   w \leftarrow w + \eta y_i x_i \\
   b \leftarrow b + \eta y_i
   $$
   式中 $ \eta(0 < \eta \le 1)$ 是步长，即学习率(learning rate).这样通过迭代可以期待损失函数 $L(w,b)$ 不断减小，直到为0.



==**算法描述**==:

​	输入：训练数据集 $T = \{(x_1,y_1),(x_2,y_2),(x_3,y_3),\dots,(x_N,y_N)\}$, 其中，$x_i \in R^n,y_i \in Y = \{+1, -1\}, i=1,2,\dots,N;$ 学习率$ \eta(0 < \eta \le 1);$ 

​	输出：$w , b;$ 感知机模型 $f(x) = sign(w\cdot x + b)$.

​	(1) 选取初值 $w_o, b_0;$
​	(2) 在训练集中选取数据 $(x_i, y_i);$
​	(3) 如果$y_i(w\cdot x_i +b) \le 0,$
$$
w \leftarrow w+ \eta y_i x_i
\\
b \leftarrow b + \eta y_i
$$
​	(4) 转至(2) , 直到训练集中没有误分类点。



### 3.2 算法的收敛性

- 将偏置 $b$ 并入权重向量 $w$ ，记作 $ \hat{w} = (w^T, b)^T$ 

- 将输入向量加以扩充，加进常数 1 , 记作 $\hat{x} = (x^T, 1)^T$

- $\hat{w} \in R^{n+1}$ , $ \hat{x} \in R^{n+1}$

- $ \hat{w}\cdot\hat{x} = w\cdot x +b $                                                  
  $$
  \hat{w}\cdot\hat{x} =
  \left[
  \begin{array}{cccc}			 
      w^T\\ 
      b\\ 
  \end{array}
  \right]
  \cdot
  \left[
  \begin{array}{cccc}			 
      x^T\\ 
      1\\ 
  \end{array}
  \right]
  =
  w \cdot x + b
  &&&&&(1)
  $$
  

==**定理(Novikoff)**==:

设训练数据集 $T = \{(x_1,y_1),(x_2,y_2),(x_3,y_3),\dots,(x_N,y_N)\}$ 是线性可分的，其中 $x_i \in X = R^n,y_i \in Y = \{+1, -1\}, i=1,2,\dots,N,$ 则

​	(1) 存在满足条件 $ ||\hat{w}_{opt} = 1||$ 的超平面 $ \hat{w} _{opt} \cdot \hat{x} = w_{opt} \cdot x + b_{opt} = 0$ 将训练数据集完全正确分开； 且存在 $\gamma > 0$, 对所有 $i=1,2,\dots,N$	
$$
y_i(\hat{w}_{opt} \cdot \hat{x}_i) = y_i(w_{opt} \cdot x_i + b_{opt}) \ge \gamma
$$
​	(2) 令 $R = \underset{1\le i \le N}{max} ||\hat{x}_i||$ , 则感知机算法3.1 在训练数据集上的误分类次数 $k $ 满足不等式(3.2)
$$
k \le (\frac{R}{\gamma})^2
$$
**不等式(3.2)证明：**

- 感知机算法从 $\hat{w}_0 = 0$ 开始的，如果实例被误分类， 则更新权重。令 $\hat{w}_{k-1}$ 是第k个误分类实例之前的扩充权重向量，即
  $$
  \hat{w}_{k-1} = (w^T_{k-1},b_{k-1} )^T
  $$
  则第k个误分类实例的条件是
  $$
  y_i(\hat{w}_{k-1} \cdot \hat{x}_i) = y_i (w_{k-1} \cdot x_i + b_{k-1}) \le 0
  $$

- 若 $(x_i, y_i)$ 是被 $\hat{w}_{k-1} = (w^T_{k-1},b_{k-1})^T$ 误分类的数据， 则 $w$ 和 $b$ 的更新是
  $$
  w_k \leftarrow w_{k-1} + \eta y_ix_i\\
  b_k \leftarrow b_{k-1} + \eta y_i
  $$
  即
  $$
  \hat{w}_k = \hat{w}_{k-1} + \eta y_i \hat{x}_i
  $$

**推导一：**

- 推导
  $$
  \hat{w}_k \cdot \hat{w}_{opt} \ge k\eta \gamma
  $$
  

​	由上述可知，
$$
\hat{w}_k \cdot \hat{w}_{opt} = \hat{w}_{k-1} \cdot \hat{w}_{opt} + \eta y_i \hat{w}_{opt} \cdot \hat{x}_i \\
\ge \hat{w}_{k-1} \cdot \hat{w}_{opt} + \eta \gamma\\
$$
​	而对于 $\gamma$ , 有 $ \gamma = \underset{i}{min}\{y_i(w_{opt} \cdot x_i + b_{opt})\} = \underset{i}{min} \{y_i ( \hat{w}_{opt} \cdot \hat{x}_i)\}$ , 故有⬆

​	由此，递推
$$
\hat{w}_{k-1} \cdot \hat{w}_{opt} + \eta \gamma= \hat{w}_{k-2} \cdot \hat{w}_{opt} + 2\eta \gamma = \dots = k\eta \gamma
$$
​	故有
$$
\hat{w}_k \cdot \hat{w}_{opt} \ge k\eta \gamma
$$


**推导二:**

- 推导
  $$
  ||\hat{w}_k||^2 \le k\eta ^2 R^2
  $$

​	

​	由$\hat{w}_k = \hat{w}_{k-1} + \eta y_i \hat{x}_i$ 可得 $||\hat{w}_k||^2 = ||\hat{w}_{k-1}||^2 + 2\eta y_i \hat{w}_{k-1} \cdot \hat{x}_i + \eta ^2||\hat{x}_i||^2$

​	而第k个误分类实例条件是 $ y_i(\hat{w}_{k-1} \cdot \hat{x}_i) = y_i (w_{k-1} \cdot x_i + b_{k-1}) \le 0$ , 且 $0 < \eta \le 1$, 因此 $2\eta y_i(\hat{w}_{k-1} \cdot \hat{x}_i) \le 0$, 故有
$$
||\hat{w}_k||^2 \le ||\hat{w}_{k-1}||^2 + \eta ^2||\hat{x}_i||^2 \le ||\hat{w}_{k-2}||^2 + 2\eta ^2||\hat{x}_i||^2 \le \dots \le  k\eta ^2||\hat{x}_i||^2
$$
​	又 $R = \underset{1\le i \le N}{max} ||\hat{x}_i|| $ , 故推出 $ ||\hat{w}_k||^2 \le k\eta ^2||\hat{x}_i||^2 \le k \eta ^2 R^2$

由==推导一== 和 ==推导二== 得：
$$
k \eta \gamma \le \hat{w}_k \cdot \hat{w}_{opt} \le ||\hat{w}_k||\cdot||\hat{w}_{opt}|| \le ||\hat{w}_k||^2 \le \sqrt{k}\eta R \\
k^2\gamma ^2 \le k R^2
$$
故有==$k \le ({\frac{R}{\gamma}})^2$==



**结论:**
	误分类的次数 $k$ 是有上界的，经过有限次搜索可以找到将训练数据完全正确分开的分离超平面。当训练数据集线性可分时，感知机学习算法原始形式迭代是收敛的。当训练集线性不可分时，感知机学习算法不收敛，迭代结果会发生震荡。







### 3.3 感知机学习算法的对偶形式

 ==**基本思想**==:

将 $w$ 和 $ b$ 表示为实例 $x_i $ 和 标记 $ y_i$ 的线性组合的形式，通过求解其系数而求得 $w$ 和 $b$ . 为不失一般性，==可假定初始值均为 0.== 则对误分类点 $(x_i, y_i)$ 通过
$$
w \leftarrow w+ \eta y_i x_i
\\
b \leftarrow b + \eta y_i
$$
逐步更新，若设置更新 n 次， 则 $w, b$ 关于$(x_i,y_i)$ 的增量分别是 $\alpha_i y_i x_i $ 和 $\alpha_i y_i$， 这里$\alpha_i = n_i \eta$, 是点 $(x_i, y_i)$ 被误分类的次数。这样，$w, b$ 可分别表示为
$$
w = \sum^N_{i=1}\alpha_i y_i x_i
\\
b =  \sum^N_{i=1}\alpha_i y_i
$$
实例点更新次数越多，意味着它距离分离超平面越近，也就越难正确分类。



==**算法描述:**==

输入：训练数据集 $T = \{(x_1,y_1),(x_2,y_2),(x_3,y_3),\dots,(x_N,y_N)\}$, 其中，$x_i \in R^n,y_i \in Y = \{+1, -1\}, i=1,2,\dots,N;$ 学习率$ \eta(0 < \eta \le 1);$ 

输出：$\alpha, b;$ 感知机模型$f(x) = sign(\sum^N_{j=1}\alpha_j y_j x_j \cdot x+b),$ 其中$\alpha = (\alpha_1,\alpha_2,\dots,\alpha_N)^T$。

​	（1）$\alpha \leftarrow 0,b \leftarrow 0;$

​	（2）在训练集中选取数据$(x_i, y_i);$

​	（3）如果$y_i(\sum^N_{j=1}\alpha_j y_j x_j \cdot x+b) \le 0,$
$$
\alpha \leftarrow \alpha + \eta
\\
b \leftarrow b + \eta y_i
$$
​	（4）转至（2）直到没有误分类数据



​	对偶形式中训练实例仅以内积的形式出现。为了方便，可以预先将训练集中实例间的内积计算出来并以矩阵的形式存储，这个矩阵就是所谓的Gram矩阵
$$
G = [x_i \cdot y_i]_{N \times N}
$$
