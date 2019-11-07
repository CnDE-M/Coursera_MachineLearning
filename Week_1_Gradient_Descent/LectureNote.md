# Machine Learning(Andrew Ng) - Week 1 Notes

<p align="center">
*Xinyue Ma   1653515@tongji.edu.cn*
</p>


## Definition

### 1. Supervised Learning
+ Dataset: 
{ Feature_1, Feature_2, ... Feature_n, Right Answer};

> i.e. we have datasets of **m**:

> $$\vec{Y} = (y^{(1)}, y^{(2)}, \cdots, y^{(m)}) ^{T}$$

> For each record i, there are features of **n**:

> $$ \vec{X} = ( a,b ) $$

>$$X^{(i)} = (1, x_1^{(i)}, x_2^{(i)}, \cdots, x_n^{(i)})^{T} \quad (i=1, 2, \cdots, m) $$

> Feature coefficiency:

> $$\vec{\theta} = (\theta_0, \theta_1, \theta_2, \cdots, \theta_n)^T $$

> linear regression model:

> $$h_\theta(x^{(i)})= \theta_0\cdot 1 + \theta_1\cdot x_1^{(i)} + \theta_2\cdot x_2^{(i)} + \cdots + \theta_n\cdot x_n^{(i)}$$

> Or

> $$ h_\theta(X) = \vec{X}\cdot\vec{\theta}$$



+ Models:
	- Discrete output: Classification;
	- Continuous output: Regression;


### 2. Cost Function
A good model will produce results $$ h_\theta(X)$$ which has the least difference with the right answer $$Y$$, 

That is to minimize the cost function $$J(\theta)$$:

$$J(\theta) = \sum^m_{i=1} (h\theta(X^{(i)})-Y^{(i)})^2$$

Below is one the the way to get optimal $$\theta$$

### 3. Gradient Descent