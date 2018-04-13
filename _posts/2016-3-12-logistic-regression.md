---
layout: default
title: Logistic Regression 模型推导整理
---


Logistic Regression 模型的假设函数定义如下：

$$  
\begin{align}  
h_{\theta}(x) & = \frac{1}{1+e^{-\theta^T x}} \\
& = \frac{e^{\theta^T x}}{1+e^{\theta^T x}} \\
\end{align}
$$  

其中， $$ \theta $$ 为参数， $$ x $$ 为样本的特征， $$ \theta $$ 和 $$ x $$ 均为列向量，假设函数可以认为是将该样本标记为正例的概率，即：

$$  
\begin{align}
P(Y=1 \mid x) & = h_{\theta}(x) = \frac{e^{\theta^T x}}{1+e^{\theta^T x}} \\
P(Y=0 \mid x) & = 1 - h_{\theta}(x) = \frac{1}{1+e^{\theta^T x}}
\end{align}
$$  

对于给定训练集(m个样本)，应用 Logistic Regression 模型学习时，我们可以用极大似然估计模型参数 $$ \theta $$ 。似然函数为

$$  
\prod_{i=1}^{m}P(Y=1 \mid x^{(i)})^{y^{(i)}} P(Y=0 \mid x^{(i)})^{1-y^{(i)}}  
$$  

即

$$  
\prod_{i=1}^{m}h_{\theta}(x^{(i)})^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}
$$  

其对数似然函数为

$$  
\begin{align}
L(\theta) & = \sum_{i=1}^{m}[y^{(i)}\log h_{\theta}(x^{(i)}) + (1-y^{(i)})\log(1-h_{\theta}(x^{(i)}))] \\
& = \sum_{i=1}^{m}[y^{(i)}(\theta^T x^{(i)}) - \log (1+e^{\theta^T x^{(i)}})] \\
\end{align}
$$  

Cost Function 可定义为

$$  
\begin{align}
J(\theta) & =\ - L(\theta) \\
& =\ - \sum_{i=1}^{m}[y^{(i)}(\theta^T x^{(i)}) - \log (1+e^{\theta^T x^{(i)}})] \\
\end{align}
$$  

那么我们现在的目标是求极大似然，也就是最小化 $$ J(\theta) $$ ，可以利用梯度下降。计算梯度的方法是对 $$ \theta_j $$ 求偏导

$$  
\begin{align}
\frac{\partial}{\partial \theta_j}J(\theta) & =\ - \sum_{i=1}^{m}[y^{(i)}x_{j}^{(i)} - \frac{x_{j}^{(i)}e^{\theta^T x^{(i)}}}{1+e^{\theta^T x^{(i)}}}]  \\
 & = \sum_{i=1}^{m} x_{j}^{(i)}(\frac{e^{\theta^T x^{(i)}}}{1+e^{\theta^T x^{(i)}}} - y^{(i)})  \\
 & = \sum_{i=1}^{m} x_{j}^{(i)}(h_{\theta}(x^{(i)}) - y^{(i)}) \\
\end{align}
$$  

即

$$  
\nabla_{\theta}J(\theta) = \sum_{i=1}^{m}x^{(i)}(h_{\theta}(x^{(i)}) - y^{(i)})
$$  

---
参考资料：  
[1]. 统计学习方法/李航著. ——北京：清华大学出版社，2012.  
[2]. <a href="http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression/" target="_blank">http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression/</a>  
[3]. <a href="https://www.coursera.org/learn/machine-learning" target="_blank">https://www.coursera.org/learn/machine-learning</a>  
