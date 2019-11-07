# Machine Learning(Andrew Ng) - Week 1 Notes

<p align="center">
Xinyue Ma   1653515@tongji.edu.cn
</p>


## Definition

### 1. Supervised Learning
+ Dataset: 
{ Feature_1, Feature_2, ... Feature_n, Right Answer};

> i.e. we have datasets of **m**:
>
><a href="https://www.codecogs.com/eqnedit.php?latex=$$\vec{Y}&space;=&space;(y^{(1)},&space;y^{(2)},&space;\cdots,&space;y^{(m)})&space;^{T}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\vec{Y}&space;=&space;(y^{(1)},&space;y^{(2)},&space;\cdots,&space;y^{(m)})&space;^{T}$$" title="$$\vec{Y} = (y^{(1)}, y^{(2)}, \cdots, y^{(m)}) ^{T}$$" /></a>
>
> For each record i, there are features of **n**:
>
<!--
\vec{X}_{m \times n}=
\left(
\begin{matrix}
 1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
 1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\
 \vdots & \vdots & \vdots & \ddots & \vdots \\
 1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\
\end{matrix}
\right)
-->
> <a href="https://www.codecogs.com/eqnedit.php?latex=\vec{X}_{m&space;\times&space;n}=&space;\left(&space;\begin{matrix}&space;1&space;&&space;x_1^{(1)}&space;&&space;x_2^{(1)}&space;&&space;\cdots&space;&&space;x_n^{(1)}&space;\\&space;1&space;&&space;x_1^{(2)}&space;&&space;x_2^{(2)}&space;&&space;\cdots&space;&&space;x_n^{(2)}&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;1&space;&&space;x_1^{(m)}&space;&&space;x_2^{(m)}&space;&&space;\cdots&space;&&space;x_n^{(m)}&space;\\&space;\end{matrix}&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\vec{X}_{m&space;\times&space;n}=&space;\left(&space;\begin{matrix}&space;1&space;&&space;x_1^{(1)}&space;&&space;x_2^{(1)}&space;&&space;\cdots&space;&&space;x_n^{(1)}&space;\\&space;1&space;&&space;x_1^{(2)}&space;&&space;x_2^{(2)}&space;&&space;\cdots&space;&&space;x_n^{(2)}&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;1&space;&&space;x_1^{(m)}&space;&&space;x_2^{(m)}&space;&&space;\cdots&space;&&space;x_n^{(m)}&space;\\&space;\end{matrix}&space;\right)" title="\vec{X}_{m \times n}= \left( \begin{matrix} 1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\ 1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\ \end{matrix} \right)" /></a>
>
><!--$$X^{(i)} = (1, x_1^{(i)}, x_2^{(i)}, \cdots, x_n^{(i)})^{T} \quad (i=1, 2, \cdots, m) $$ -->
<a href="https://www.codecogs.com/eqnedit.php?latex=$$X^{(i)}&space;=&space;(1,&space;x_1^{(i)},&space;x_2^{(i)},&space;\cdots,&space;x_n^{(i)})^{T}&space;\quad&space;(i=1,&space;2,&space;\cdots,&space;m)&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$X^{(i)}&space;=&space;(1,&space;x_1^{(i)},&space;x_2^{(i)},&space;\cdots,&space;x_n^{(i)})^{T}&space;\quad&space;(i=1,&space;2,&space;\cdots,&space;m)&space;$$" title="$$X^{(i)} = (1, x_1^{(i)}, x_2^{(i)}, \cdots, x_n^{(i)})^{T} \quad (i=1, 2, \cdots, m) $$" /></a>
>
> Feature coefficiency:
>
> <!--$$\vec{\theta} = (\theta_0, \theta_1, \theta_2, \cdots, \theta_n)^T $$-->
><a href="https://www.codecogs.com/eqnedit.php?latex=$$\vec{\theta}&space;=&space;(\theta_0,&space;\theta_1,&space;\theta_2,&space;\cdots,&space;\theta_n)^T&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\vec{\theta}&space;=&space;(\theta_0,&space;\theta_1,&space;\theta_2,&space;\cdots,&space;\theta_n)^T&space;$$" title="$$\vec{\theta} = (\theta_0, \theta_1, \theta_2, \cdots, \theta_n)^T $$" /></a>
> 
> The linear regression model:
>
> <!--$$h_\theta(x^{(i)})= \theta_0\cdot 1 + \theta_1\cdot x_1^{(i)} + \theta_2\cdot x_2^{(i)} + \cdots + \theta_n\cdot x_n^{(i)}$$-->
><a href="https://www.codecogs.com/eqnedit.php?latex=$$h_\theta(x^{(i)})=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;x_1^{(i)}&space;&plus;&space;\theta_2\cdot&space;x_2^{(i)}&space;&plus;&space;\cdots&space;&plus;&space;\theta_n\cdot&space;x_n^{(i)}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$h_\theta(x^{(i)})=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;x_1^{(i)}&space;&plus;&space;\theta_2\cdot&space;x_2^{(i)}&space;&plus;&space;\cdots&space;&plus;&space;\theta_n\cdot&space;x_n^{(i)}$$" title="$$h_\theta(x^{(i)})= \theta_0\cdot 1 + \theta_1\cdot x_1^{(i)} + \theta_2\cdot x_2^{(i)} + \cdots + \theta_n\cdot x_n^{(i)}$$" /></a>
> Or
>
> <!--$$ h_\theta(\vec{X}) = \vec{X}\cdot\vec{\theta}$$-->
<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;h_\theta(\vec{X})&space;=&space;\vec{X}\cdot\vec{\theta}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;h_\theta(\vec{X})&space;=&space;\vec{X}\cdot\vec{\theta}$$" title="$$ h_\theta(\vec{X}) = \vec{X}\cdot\vec{\theta}$$" /></a>



+ Models:
	- Discrete output: Classification;
	- Continuous output: Regression;


### 2. Cost Function
A good model will produce results h<sub>θ</sub> which has the least difference with the right answer Y, 

That is to minimize the cost function J(θ):

<!--$$J(\theta) = \sum^m_{i=1} (h_\theta(X^{(i)})-Y^{(i)})^2$$-->
<a href="https://www.codecogs.com/eqnedit.php?latex=$$J(\theta)&space;=&space;\sum^m_{i=1}&space;(h_\theta(X^{(i)})-Y^{(i)})^2$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$J(\theta)&space;=&space;\sum^m_{i=1}&space;(h_\theta(X^{(i)})-Y^{(i)})^2$$" title="$$J(\theta) = \sum^m_{i=1} (h_\theta(X^{(i)})-Y^{(i)})^2$$" /></a>

Below is one the the way to get optimal θ

### 3. Gradient Descent
