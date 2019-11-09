# Machine Learning(Andrew Ng) - Week 1 Notes

<p align="center">
Xinyue Ma   1653515@tongji.edu.cn
</p>

## Overview
+ definition of "supervised learning"
+ principle of "gradient descent"
+ principle of "normal equation"

## Definition

### 1. Supervised Learning
+ Dataset: 
{ Feature_1, Feature_2, ... Feature_n, Right Answer};

> i.e. we have datasets of **m**:
>
> <!--$$Y_{m \times 1} =(y^{(1)}, y^{(2)}, \cdots, y^{(m)})^{T}$$-->
>
><p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$$Y_{m&space;\times&space;1}&space;=(y^{(1)},&space;y^{(2)},&space;\cdots,&space;y^{(m)})^{T}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$Y_{m&space;\times&space;1}&space;=(y^{(1)},&space;y^{(2)},&space;\cdots,&space;y^{(m)})^{T}$$" title="$$Y_{m \times 1} =(y^{(1)}, y^{(2)}, \cdots, y^{(m)})^{T}$$" /></a></p>
>
> For each record i, there are features of **n**:
>
> <!--$$\vec{x}^{(i)} = (1, x_1^{(i)}, x_2^{(i)}, \cdots, x_n^{(i)})^{T} \quad (i=1, 2, \cdots, m) $$-->
>
> <p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$$\vec{x}^{(i)}&space;=&space;(1,&space;x_1^{(i)},&space;x_2^{(i)},&space;\cdots,&space;x_n^{(i)})^{T}&space;\quad&space;(i=1,&space;2,&space;\cdots,&space;m)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\vec{x}^{(i)}&space;=&space;(1,&space;x_1^{(i)},&space;x_2^{(i)},&space;\cdots,&space;x_n^{(i)})^{T}&space;\quad&space;(i=1,&space;2,&space;\cdots,&space;m)" title="$$\vec{x}^{(i)} = (1, x_1^{(i)}, x_2^{(i)}, \cdots, x_n^{(i)})^{T} \quad (i=1, 2, \cdots, m)" /></a></p>
>
> <!--X_{m \times n}=\left(\begin{matrix}1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\\vdots & \vdots & \vdots & \ddots & \vdots \\1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\ \end{matrix}\right)-->
>
> <p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=X_{m&space;\times&space;n}=\left(\begin{matrix}1&space;&&space;x_1^{(1)}&space;&&space;x_2^{(1)}&space;&&space;\cdots&space;&&space;x_n^{(1)}&space;\\1&space;&&space;x_1^{(2)}&space;&&space;x_2^{(2)}&space;&&space;\cdots&space;&&space;x_n^{(2)}&space;\\\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\1&space;&&space;x_1^{(m)}&space;&&space;x_2^{(m)}&space;&&space;\cdots&space;&&space;x_n^{(m)}&space;\\&space;\end{matrix}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{m&space;\times&space;n}=\left(\begin{matrix}1&space;&&space;x_1^{(1)}&space;&&space;x_2^{(1)}&space;&&space;\cdots&space;&&space;x_n^{(1)}&space;\\1&space;&&space;x_1^{(2)}&space;&&space;x_2^{(2)}&space;&&space;\cdots&space;&&space;x_n^{(2)}&space;\\\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\1&space;&&space;x_1^{(m)}&space;&&space;x_2^{(m)}&space;&&space;\cdots&space;&&space;x_n^{(m)}&space;\\&space;\end{matrix}\right)" title="X_{m \times n}=\left(\begin{matrix}1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\\vdots & \vdots & \vdots & \ddots & \vdots \\1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\ \end{matrix}\right)" /></a></p>
>
>  
> Feature coefficiency:
>
> <!--$$\vec{\theta} = (\theta_0, \theta_1, \theta_2, \cdots, \theta_n)^T $$-->
>
><p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$$\vec{\theta}&space;=&space;(\theta_0,&space;\theta_1,&space;\theta_2,&space;\cdots,&space;\theta_n)^T&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\vec{\theta}&space;=&space;(\theta_0,&space;\theta_1,&space;\theta_2,&space;\cdots,&space;\theta_n)^T&space;$$" title="$$\vec{\theta} = (\theta_0, \theta_1, \theta_2, \cdots, \theta_n)^T $$" /></a></p>
> 
> The linear regression model:
>
> <!--$$h_\theta(x^{(i)})= \theta_0\cdot 1 + \theta_1\cdot x_1^{(i)} + \theta_2\cdot x_2^{(i)} + \cdots + \theta_n\cdot x_n^{(i)}$$-->
>
> <p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$$h_\theta(x^{(i)})=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;x_1^{(i)}&space;&plus;&space;\theta_2\cdot&space;x_2^{(i)}&space;&plus;&space;\cdots&space;&plus;&space;\theta_n\cdot&space;x_n^{(i)}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$h_\theta(x^{(i)})=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;x_1^{(i)}&space;&plus;&space;\theta_2\cdot&space;x_2^{(i)}&space;&plus;&space;\cdots&space;&plus;&space;\theta_n\cdot&space;x_n^{(i)}$$" title="$$h_\theta(x^{(i)})= \theta_0\cdot 1 + \theta_1\cdot x_1^{(i)} + \theta_2\cdot x_2^{(i)} + \cdots + \theta_n\cdot x_n^{(i)}$$" /></a></p>
>
> Or
>
> <!--$$ h_\theta(X) = X\cdot\vec{\theta}$$-->
>
>  <p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;h_\theta(X)&space;=&space;X\cdot\vec{\theta}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;h_\theta(X)&space;=&space;X\cdot\vec{\theta}$$" title="$$ h_\theta(X) = X\cdot\vec{\theta}$$" /></a></p>


+ Models:
	- Discrete output: **Classification**;
	- Continuous output: **Regression**;


### 2. Cost Function
A good model will produce results h<sub>θ</sub> which has the least difference with the right answer Y, 

That is to minimize the cost function J(θ):

<!--$$J(\theta) = \sum^m_{i=1} (h_\theta(\vec{x}^{(i)})-Y^{(i)})^2$$-->

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$$J(\theta)&space;=&space;\sum^m_{i=1}&space;(h_\theta(\vec{x}^{(i)})-Y^{(i)})^2$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$J(\theta)&space;=&space;\sum^m_{i=1}&space;(h_\theta(\vec{x}^{(i)})-Y^{(i)})^2$$" title="$$J(\theta) = \sum^m_{i=1} (h_\theta(\vec{x}^{(i)})-Y^{(i)})^2$$" /></a></p>

Below is one of the way to get optimal θ

### 3. Gradient Descent

This is to find θ, so that 

<!--\frac{\partial J(\theta)}{\partial \theta_j} = 0, \quad j = 1,2, ..., n-->

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;J(\theta)}{\partial&space;\theta_j}&space;=&space;0,&space;\quad&space;j&space;=&space;1,2,&space;...,&space;n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J(\theta)}{\partial&space;\theta_j}&space;=&space;0,&space;\quad&space;j&space;=&space;1,2,&space;...,&space;n" title="\frac{\partial J(\theta)}{\partial \theta_j} = 0, \quad j = 1,2, ..., n" /></a></p>

"Gradient descent" is to simutaneously change each θ<sub>j</sub>'s value along with their partial derivative, iterate until all partial derivative equals to zero (or a weaken condition, the cost function value less than a threshold).

<!--\theta_j' = \theta_j - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta_j}, \quad j = 1,2, ..., n-->

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\theta_j'&space;=&space;\theta_j&space;-&space;\alpha&space;\cdot&space;\frac{\partial&space;J(\theta)}{\partial&space;\theta_j},&space;\quad&space;j&space;=&space;1,2,&space;...,&space;n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_j'&space;=&space;\theta_j&space;-&space;\alpha&space;\cdot&space;\frac{\partial&space;J(\theta)}{\partial&space;\theta_j},&space;\quad&space;j&space;=&space;1,2,&space;...,&space;n" title="\theta_j' = \theta_j - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta_j}, \quad j = 1,2, ..., n" /></a></p>

In the above equation, __"α"__ determines step length (constant), __"partial derivative"__ determine both step length (decrease when proning to 0) and the orientation (increase to 0 or decrease to 0).
For a large positive partial derivative, θ will decrease fast (imaging a car sliding fast at a steep slope); and for a small negative partial derivative, θ will increase slow (imaging a car sliding fast at a steep slope), they are all inclined to zero. **This could explain only one constant α is required in the equation.**

<div align=center>
	<img width="500" height="285" src="https://github.com/CnDE-M/Coursera_MarchineLearning/blob/master/Week_1_Gradient_Descent/svgs/gradient_descent.png"/>
</div>

Below is an actual gradient descent path of (θ0, θ1)~cost schematic image I made:
<div align=center>
	<img width="500" height="285" src="https://github.com/CnDE-M/Coursera_MachineLearning/blob/master/Week_1_Gradient_Descent/svgs/gradient%20descent%20path.png"/>
</div>


Let's simplify the equation:
<!--$$\frac{\partial{J(\vec{\theta})}}{\partial{\theta_j}} = 2 \times \theta_j \times \sum^m_{i=1} (h_\theta(\vec{x}^{(i)})-Y^{(i)}), \quad j=1, 2, \cdots. n $$-->

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$$\frac{\partial{J(\vec{\theta})}}{\partial{\theta_j}}&space;=&space;2&space;\times&space;\theta_j&space;\times&space;\sum^m_{i=1}&space;(h_\theta(\vec{x}^{(i)})-Y^{(i)}),&space;\quad&space;j=1,&space;2,&space;\cdots.&space;n&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\frac{\partial{J(\vec{\theta})}}{\partial{\theta_j}}&space;=&space;2&space;\times&space;\theta_j&space;\times&space;\sum^m_{i=1}&space;(h_\theta(\vec{x}^{(i)})-Y^{(i)}),&space;\quad&space;j=1,&space;2,&space;\cdots.&space;n&space;$$" title="$$\frac{\partial{J(\vec{\theta})}}{\partial{\theta_j}} = 2 \times \theta_j \times \sum^m_{i=1} (h_\theta(\vec{x}^{(i)})-Y^{(i)}), \quad j=1, 2, \cdots. n $$" /></a></p>

To formalize the simplyify the equation, 1/2m is multipled in the front, and the θ gradient descent is:

<!--$$\theta_j' = \theta_j - \alpha \times \frac{\theta_j}{m} \times \sum^m_{i=1} (h_\theta(\vec{x}^{(i)})-Y^{(i)}), \quad j=1, 2, \cdots. n $$-->

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$$\theta_j'&space;=&space;\theta_j&space;-&space;\alpha&space;\times&space;\frac{\theta_j}{m}&space;\times&space;\sum^m_{i=1}&space;(h_\theta(\vec{x}^{(i)})-Y^{(i)}),&space;\quad&space;j=1,&space;2,&space;\cdots.&space;n&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\theta_j'&space;=&space;\theta_j&space;-&space;\alpha&space;\times&space;\frac{\theta_j}{m}&space;\times&space;\sum^m_{i=1}&space;(h_\theta(\vec{x}^{(i)})-Y^{(i)}),&space;\quad&space;j=1,&space;2,&space;\cdots.&space;n&space;$$" title="$$\theta_j' = \theta_j - \alpha \times \frac{\theta_j}{m} \times \sum^m_{i=1} (h_\theta(\vec{x}^{(i)})-Y^{(i)}), \quad j=1, 2, \cdots. n $$" /></a></p>

### 4. Tips

1. Prepossess data

+ feature scaling

<!--$$x_i' = \frac{x_i}{x_{max}-x_{min}}$$-->

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=x_i'&space;=&space;\frac{x_i}{x_{max}-x_{min}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i'&space;=&space;\frac{x_i}{x_{max}-x_{min}}" title="x_i' = \frac{x_i}{x_{max}-x_{min}}" /></a></p>

+ mean normalization

<!--$$ x' = \frac{x - \mu}{\delta} $$-->

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=x'&space;=&space;\frac{x&space;-&space;\mu}{\delta}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x'&space;=&space;\frac{x&space;-&space;\mu}{\delta}" title="x' = \frac{x - \mu}{\delta}" /></a></p>

2. plot out J(θ)~iteration time
Check if J(θ) decrease with the increase of iteration time, this will lead you find a proper α。

<div align=center>
	<img width="500" height="285" src="https://github.com/CnDE-M/Coursera_MarchineLearning/blob/master/Week_1_Gradient_Descent/svgs/gradient_descent_3.png"/>
</div>

3. Polymonial regression model
Simply treat polymonial vaiable as a new feature.

<!--$$ h_\theta(x)= \theta_0\cdot 1 + \theta_1\cdot x + \theta_2\cdot x^2 + \theta_3\cdot x^3  $$-->

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=h_\theta(x)=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;x&space;&plus;&space;\theta_2\cdot&space;x^2&space;&plus;&space;\theta_3\cdot&space;x^3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_\theta(x)=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;x&space;&plus;&space;\theta_2\cdot&space;x^2&space;&plus;&space;\theta_3\cdot&space;x^3" title="h_\theta(x)= \theta_0\cdot 1 + \theta_1\cdot x + \theta_2\cdot x^2 + \theta_3\cdot x^3" /></a></p>

equals to:
<!--$$ h_\theta(t)= \theta_0\cdot 1 + \theta_1\cdot t_1 + \theta_2\cdot t_2 + \theta_3\cdot t_3, \quad$$ \begin{cases} t_1 = x \\t_2 = x^2\\t_3 = x^3\end{cases}-->

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;h_\theta(t)=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;t_1&space;&plus;&space;\theta_2\cdot&space;t_2&space;&plus;&space;\theta_3\cdot&space;t_3,&space;\quad$$&space;\begin{cases}&space;t_1&space;=&space;x&space;\\&space;t_2&space;=&space;x^2\\&space;t_3&space;=&space;x^3&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;h_\theta(t)=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;t_1&space;&plus;&space;\theta_2\cdot&space;t_2&space;&plus;&space;\theta_3\cdot&space;t_3,&space;\quad$$&space;\begin{cases}&space;t_1&space;=&space;x&space;\\&space;t_2&space;=&space;x^2\\&space;t_3&space;=&space;x^3&space;\end{cases}" title="$$ h_\theta(t)= \theta_0\cdot 1 + \theta_1\cdot t_1 + \theta_2\cdot t_2 + \theta_3\cdot t_3, \quad$$ \begin{cases} t_1 = x \\ t_2 = x^2\\ t_3 = x^3 \end{cases}" /></a></p>



### Normal Equation
(The online course doesn't cover why the equation is, here shows the deduction.)

The question could be regarded as a "norm optimazation problem".

<!--min\left \|  \vec{x} \right \|_p = \sum \sqrt[0]{\left | x \right |^p}\quad s.t. \quad A \times \vec{x} = \vec{b}-->

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=min\left&space;\|&space;\vec{x}&space;\right&space;\|_p&space;=&space;\sum&space;\sqrt[0]{\left&space;|&space;x&space;\right&space;|^p}\quad&space;s.t.&space;\quad&space;A&space;\times&space;\vec{x}&space;=&space;\vec{b}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min\left&space;\|&space;\vec{x}&space;\right&space;\|_p&space;=&space;\sum&space;\sqrt[0]{\left&space;|&space;x&space;\right&space;|^p}\quad&space;s.t.&space;\quad&space;A&space;\times&space;\vec{x}&space;=&space;\vec{b}" title="min\left \| \vec{x} \right \|_p = \sum \sqrt[0]{\left | x \right |^p}\quad s.t. \quad A \times \vec{x} = \vec{b}" /></a></p>


+ L-0 norm optimazation returns mininal X number, however, its math expression is hard to express;
+ L-1 norm optimazation returns mininal X sum value, it will lead to sparse result (**lasso regression**);
+ L-2 norm optimazation returns mininal X^2 sum value, it will not sparse, but keep all features, spreading influence to all, this would help if every features did has but not big contribute to the result(**ridge regression**).

The linear regression model is can be solved by L-2 optimazation

<!--min\left \|  \vec{\theta} \right \|_2 = \sum_{j-1}^n \sqrt[2]{\left | \theta_j \right |^2} \quad s.t. \quad  X_{m\times n} \times \vec{\theta}=\vec{Y}_{m\times 1}-->

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=min\left&space;\|&space;\vec{\theta}&space;\right&space;\|_2&space;=&space;\sum_{j-1}^n&space;\sqrt[2]{\left&space;|&space;\theta_j&space;\right&space;|^2}&space;\quad&space;s.t.&space;\quad&space;X_{m\times&space;n}&space;\times&space;\vec{\theta}=\vec{Y}_{m\times&space;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min\left&space;\|&space;\vec{\theta}&space;\right&space;\|_2&space;=&space;\sum_{j-1}^n&space;\sqrt[2]{\left&space;|&space;\theta_j&space;\right&space;|^2}&space;\quad&space;s.t.&space;\quad&space;X_{m\times&space;n}&space;\times&space;\vec{\theta}=\vec{Y}_{m\times&space;1}" title="min\left \| \vec{\theta} \right \|_2 = \sum_{j-1}^n \sqrt[2]{\left | \theta_j \right |^2} \quad s.t. \quad X_{m\times n} \times \vec{\theta}=\vec{Y}_{m\times 1}" /></a></p>

To solve the problem, 2 points are required:
+ Lagrance Multiplier Methods
>
><!--min/max f(x_1, x_2, \cdots, x_n) \quad s.t. \quad g(x_1, x_2, \cdots, x_m) = c -->
>
> The format goes like this (c is a constant):
>
><p align="center"> <a href="https://www.codecogs.com/eqnedit.php?latex=min/max&space;f(x_1,&space;x_2,&space;\cots,&space;x_n)&space;\quad&space;s.t.&space;\quad&space;g(x_1,&space;x_2,&space;\cots,&space;x_n)&space;=&space;c" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min/max&space;f(x_1,&space;x_2,&space;\cots,&space;x_n)&space;\quad&space;s.t.&space;\quad&space;g(x_1,&space;x_2,&space;\cots,&space;x_n)&space;=&space;c" title="min/max f(x_1, x_2, \cdots, x_n) \quad s.t. \quad g(x_1, x_2, \cdots, x_m) = c" /></a></p>
>
> To solve the question, construct F(X,λ)
>
><!--F(x_1, x_2, \cdots, x_n, \lambda) = f(x_1, x_2, \cdots, x_n) + \lambda \cdot [g(x_1, x_2, \cdots, x_m)-c] =0-->
><p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=F(x_1,&space;x_2,&space;\cdots,&space;x_n,&space;\lambda)&space;=&space;f(x_1,&space;x_2,&space;\cdots,&space;x_n)&space;&plus;&space;\lambda&space;\cdot&space;[g(x_1,&space;x_2,&space;\cdots,&space;x_m)-c]&space;=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F(x_1,&space;x_2,&space;\cdots,&space;x_n,&space;\lambda)&space;=&space;f(x_1,&space;x_2,&space;\cdots,&space;x_n)&space;&plus;&space;\lambda&space;\cdot&space;[g(x_1,&space;x_2,&space;\cdots,&space;x_m)-c]&space;=0" title="F(x_1, x_2, \cdots, x_n, \lambda) = f(x_1, x_2, \cdots, x_n) + \lambda \cdot [g(x_1, x_2, \cdots, x_m)-c] =0" /></a></p>
>

Then solve all partial derivative to be equal to 0:

<!--\begin{cases}\frac{\partial{F}}{\partial{x_1}} = 0 \\
\frac{\partial{F}}{\partial{x_2}} = 0 \\
\cdots \\
\frac{\partial{F}}{\partial{x_n}} = 0 \\
\frac{\partial{F}}{\partial{\lambda}} = 0 \\
\end{cases}-->

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\begin{cases}\frac{\partial{F}}{\partial{x_1}}&space;=&space;0&space;\\&space;\frac{\partial{F}}{\partial{x_2}}&space;=&space;0&space;\\&space;\cdots&space;\\&space;\frac{\partial{F}}{\partial{x_n}}&space;=&space;0&space;\\&space;\frac{\partial{F}}{\partial{\lambda}}&space;=&space;0&space;\\&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{cases}\frac{\partial{F}}{\partial{x_1}}&space;=&space;0&space;\\&space;\frac{\partial{F}}{\partial{x_2}}&space;=&space;0&space;\\&space;\cdots&space;\\&space;\frac{\partial{F}}{\partial{x_n}}&space;=&space;0&space;\\&space;\frac{\partial{F}}{\partial{\lambda}}&space;=&space;0&space;\\&space;\end{cases}" title="\begin{cases}\frac{\partial{F}}{\partial{x_1}} = 0 \\ \frac{\partial{F}}{\partial{x_2}} = 0 \\ \cdots \\ \frac{\partial{F}}{\partial{x_n}} = 0 \\ \frac{\partial{F}}{\partial{\lambda}} = 0 \\ \end{cases}" /></a></p>

eliminate the λ and resolve all x.

+ matrix and norm derivative

Here shows deduction for normal equation:

Question:

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=min\left&space;\|&space;\vec{\theta}&space;\right&space;\|_2&space;=&space;\sum_{j-1}^n&space;\sqrt[2]{\left&space;|&space;\theta_j&space;\right&space;|^2}&space;\quad&space;s.t.&space;\quad&space;X_{m\times&space;n}&space;\times&space;\vec{\theta}=\vec{Y}_{m\times&space;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?min\left&space;\|&space;\vec{\theta}&space;\right&space;\|_2&space;=&space;\sum_{j-1}^n&space;\sqrt[2]{\left&space;|&space;\theta_j&space;\right&space;|^2}&space;\quad&space;s.t.&space;\quad&space;X_{m\times&space;n}&space;\times&space;\vec{\theta}=\vec{Y}_{m\times&space;1}" title="min\left \| \vec{\theta} \right \|_2 = \sum_{j-1}^n \sqrt[2]{\left | \theta_j \right |^2} \quad s.t. \quad X_{m\times n} \times \vec{\theta}=\vec{Y}_{m\times 1}" /></a></p>

Solution:

<!--F(\vec{\theta}) = \left \| \vec{\theta} \right \|_2 + \lambda \cdot (X \times \vec{\theta} - \vec{Y})-->

<a href="https://www.codecogs.com/eqnedit.php?latex=F(\vec{\theta})&space;=&space;\left&space;\|&space;\vec{\theta}&space;\right&space;\|_2&space;&plus;&space;\lambda&space;\cdot&space;(X&space;\times&space;\vec{\theta}&space;-&space;\vec{Y})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F(\vec{\theta})&space;=&space;\left&space;\|&space;\vec{\theta}&space;\right&space;\|_2&space;&plus;&space;\lambda&space;\cdot&space;(X&space;\times&space;\vec{\theta}&space;-&space;\vec{Y})" title="F(\vec{\theta}) = \left \| \vec{\theta} \right \|_2 + \lambda \cdot (X \times \vec{\theta} - \vec{Y})" /></a>

<!--\begin{cases}
\frac{\partial{F}}{\partial{\vec{\theta}}} =0 \\
\frac{\partial{F}}{\lambda} = 0
\end{cases}-->
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{cases}&space;\frac{\partial{F}}{\partial{\vec{\theta}}}&space;=0&space;\\&space;\frac{\partial{F}}{\lambda}&space;=&space;0&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{cases}&space;\frac{\partial{F}}{\partial{\vec{\theta}}}&space;=0&space;\\&space;\frac{\partial{F}}{\lambda}&space;=&space;0&space;\end{cases}" title="\begin{cases} \frac{\partial{F}}{\partial{\vec{\theta}}} =0 \\ \frac{\partial{F}}{\lambda} = 0 \end{cases}" /></a>



<!--\begin{cases}
F = \vec{\theta}^T \cdot \vec{\theta} - \lambda \cdot (X \cdot \vec{\theta} - \vec{Y}))\\
dF = (d\vec{\theta}^T \cdot \vec{\theta}) + (d\vec{\theta} \cdot \vec{\theta}^T) + \lambda \cdot X \cdot d\vec{\theta}\\
dF = tr(dF) \\
\end{cases}-->
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{cases}&space;F&space;=&space;\vec{\theta}^T&space;\cdot&space;\vec{\theta}&space;-&space;\lambda&space;\cdot&space;(X&space;\cdot&space;\vec{\theta}&space;-&space;\vec{Y}))\\&space;dF&space;=&space;(d\vec{\theta}^T&space;\cdot&space;\vec{\theta})&space;&plus;&space;(d\vec{\theta}&space;\cdot&space;\vec{\theta}^T)&space;&plus;&space;\lambda&space;\cdot&space;X&space;\cdot&space;d\vec{\theta}\\&space;dF&space;=&space;tr(dF)&space;\\&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{cases}&space;F&space;=&space;\vec{\theta}^T&space;\cdot&space;\vec{\theta}&space;-&space;\lambda&space;\cdot&space;(X&space;\cdot&space;\vec{\theta}&space;-&space;\vec{Y}))\\&space;dF&space;=&space;(d\vec{\theta}^T&space;\cdot&space;\vec{\theta})&space;&plus;&space;(d\vec{\theta}&space;\cdot&space;\vec{\theta}^T)&space;&plus;&space;\lambda&space;\cdot&space;X&space;\cdot&space;d\vec{\theta}\\&space;dF&space;=&space;tr(dF)&space;\\&space;\end{cases}" title="\begin{cases} F = \vec{\theta}^T \cdot \vec{\theta} - \lambda \cdot (X \cdot \vec{\theta} - \vec{Y}))\\ dF = (d\vec{\theta}^T \cdot \vec{\theta}) + (d\vec{\theta} \cdot \vec{\theta}^T) + \lambda \cdot X \cdot d\vec{\theta}\\ dF = tr(dF) \\ \end{cases}" /></a>

<!--dF = tr(dF)\\
= tr( \quad (d\vec{\theta}^T \cdot \vec{\theta}) + (d\vec{\theta} \cdot \vec{\theta}^T) + \lambda \cdot X \cdot d\vec{\theta} \quad) \\
= tr( \quad 2 \cdot \vec{\theta}^T \cdot d\vec{\theta} - \lambda \cdot X \cdot d\vec{\theta} \quad)\\
= tr( \quad(2 \cdot \vec{\theta}^T  + \lambda \cdot X )\cdot d\vec{\theta} \quad)  \\
= tr( \quad(2 \cdot \vec{\theta}  +  \lambda \cdot X^T )^T \cdot d\vec{\theta} \quad)
-->
<a href="https://www.codecogs.com/eqnedit.php?latex=dF&space;=&space;tr(dF)\\&space;=&space;tr(&space;\quad&space;(d\vec{\theta}^T&space;\cdot&space;\vec{\theta})&space;&plus;&space;(d\vec{\theta}&space;\cdot&space;\vec{\theta}^T)&space;&plus;&space;\lambda&space;\cdot&space;X&space;\cdot&space;d\vec{\theta}&space;\quad)&space;\\&space;=&space;tr(&space;\quad&space;2&space;\cdot&space;\vec{\theta}^T&space;\cdot&space;d\vec{\theta}&space;-&space;\lambda&space;\cdot&space;X&space;\cdot&space;d\vec{\theta}&space;\quad)\\&space;=&space;tr(&space;\quad(2&space;\cdot&space;\vec{\theta}^T&space;&plus;&space;\lambda&space;\cdot&space;X&space;)\cdot&space;d\vec{\theta}&space;\quad)&space;\\&space;=&space;tr(&space;\quad(2&space;\cdot&space;\vec{\theta}&space;&plus;&space;\lambda&space;\cdot&space;X^T&space;)^T&space;\cdot&space;d\vec{\theta}&space;\quad)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?dF&space;=&space;tr(dF)\\&space;=&space;tr(&space;\quad&space;(d\vec{\theta}^T&space;\cdot&space;\vec{\theta})&space;&plus;&space;(d\vec{\theta}&space;\cdot&space;\vec{\theta}^T)&space;&plus;&space;\lambda&space;\cdot&space;X&space;\cdot&space;d\vec{\theta}&space;\quad)&space;\\&space;=&space;tr(&space;\quad&space;2&space;\cdot&space;\vec{\theta}^T&space;\cdot&space;d\vec{\theta}&space;-&space;\lambda&space;\cdot&space;X&space;\cdot&space;d\vec{\theta}&space;\quad)\\&space;=&space;tr(&space;\quad(2&space;\cdot&space;\vec{\theta}^T&space;&plus;&space;\lambda&space;\cdot&space;X&space;)\cdot&space;d\vec{\theta}&space;\quad)&space;\\&space;=&space;tr(&space;\quad(2&space;\cdot&space;\vec{\theta}&space;&plus;&space;\lambda&space;\cdot&space;X^T&space;)^T&space;\cdot&space;d\vec{\theta}&space;\quad)" title="dF = tr(dF)\\ = tr( \quad (d\vec{\theta}^T \cdot \vec{\theta}) + (d\vec{\theta} \cdot \vec{\theta}^T) + \lambda \cdot X \cdot d\vec{\theta} \quad) \\ = tr( \quad 2 \cdot \vec{\theta}^T \cdot d\vec{\theta} - \lambda \cdot X \cdot d\vec{\theta} \quad)\\ = tr( \quad(2 \cdot \vec{\theta}^T + \lambda \cdot X )\cdot d\vec{\theta} \quad) \\ = tr( \quad(2 \cdot \vec{\theta} + \lambda \cdot X^T )^T \cdot d\vec{\theta} \quad)" /></a>


<!--\because \quad df = tr( \frac{\partial{F} }{\partial{\theta_j}}^T\cdot d\vec{\theta})-->
<a href="https://www.codecogs.com/eqnedit.php?latex=\because&space;\quad&space;df&space;=&space;tr(&space;\frac{\partial{F}&space;}{\partial{\theta_j}}^T\cdot&space;d\vec{\theta})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\because&space;\quad&space;df&space;=&space;tr(&space;\frac{\partial{F}&space;}{\partial{\theta_j}}^T\cdot&space;d\vec{\theta})" title="\because \quad df = tr( \frac{\partial{F} }{\partial{\theta_j}}^T\cdot d\vec{\theta})" /></a>


<!--\therefore \quad \frac{\partial{F} }{\partial{\theta_j}}^T = 
 2 \cdot \vec{\theta}  - \lambda \cdot X^T  \equiv 0-->
<a href="https://www.codecogs.com/eqnedit.php?latex=\therefore&space;\quad&space;\frac{\partial{F}&space;}{\partial{\theta_j}}^T&space;=&space;2&space;\cdot&space;\vec{\theta}&space;-&space;\lambda&space;\cdot&space;X^T&space;\equiv&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\therefore&space;\quad&space;\frac{\partial{F}&space;}{\partial{\theta_j}}^T&space;=&space;2&space;\cdot&space;\vec{\theta}&space;-&space;\lambda&space;\cdot&space;X^T&space;\equiv&space;0" title="\therefore \quad \frac{\partial{F} }{\partial{\theta_j}}^T = 2 \cdot \vec{\theta} - \lambda \cdot X^T \equiv 0" /></a></p>


<!--\therefore \lambda = -2 \cdot (X^T \cdot X)^{-1} \cdot \vec{Y} -->
<a href="https://www.codecogs.com/eqnedit.php?latex=\therefore&space;\lambda&space;=&space;-2&space;\cdot&space;(X^T&space;\cdot&space;X)^{-1}&space;\cdot&space;\vec{Y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\therefore&space;\lambda&space;=&space;-2&space;\cdot&space;(X^T&space;\cdot&space;X)^{-1}&space;\cdot&space;\vec{Y}" title="\therefore \lambda = -2 \cdot (X^T \cdot X)^{-1} \cdot \vec{Y}" /></a>


<!--\therefore \quad \vec{\theta} = (X^T \cdot X)^{-1} \cdot X^T \cdot  \vec{Y} -->
<a href="https://www.codecogs.com/eqnedit.php?latex=\therefore&space;\quad&space;\vec{\theta}&space;=&space;(X^T&space;\cdot&space;X)^{-1}&space;\cdot&space;X^T&space;\cdot&space;\vec{Y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\therefore&space;\quad&space;\vec{\theta}&space;=&space;(X^T&space;\cdot&space;X)^{-1}&space;\cdot&space;X^T&space;\cdot&space;\vec{Y}" title="\therefore \quad \vec{\theta} = (X^T \cdot X)^{-1} \cdot X^T \cdot \vec{Y}" /></a>
