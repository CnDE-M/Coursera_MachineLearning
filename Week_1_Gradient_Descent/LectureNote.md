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
> <!--$$Y_{m \times 1} =(y^{(1)}, y^{(2)}, \cdots, y^{(m)})^{T}$$-->
>
><a href="https://www.codecogs.com/eqnedit.php?latex=$$Y_{m&space;\times&space;1}&space;=(y^{(1)},&space;y^{(2)},&space;\cdots,&space;y^{(m)})^{T}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$Y_{m&space;\times&space;1}&space;=(y^{(1)},&space;y^{(2)},&space;\cdots,&space;y^{(m)})^{T}$$" title="$$Y_{m \times 1} =(y^{(1)}, y^{(2)}, \cdots, y^{(m)})^{T}$$" /></a>
>
> For each record i, there are features of **n**:
>
> <!--$$\vec{x}^{(i)} = (1, x_1^{(i)}, x_2^{(i)}, \cdots, x_n^{(i)})^{T} \quad (i=1, 2, \cdots, m) $$-->
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=$$\vec{x}^{(i)}&space;=&space;(1,&space;x_1^{(i)},&space;x_2^{(i)},&space;\cdots,&space;x_n^{(i)})^{T}&space;\quad&space;(i=1,&space;2,&space;\cdots,&space;m)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\vec{x}^{(i)}&space;=&space;(1,&space;x_1^{(i)},&space;x_2^{(i)},&space;\cdots,&space;x_n^{(i)})^{T}&space;\quad&space;(i=1,&space;2,&space;\cdots,&space;m)" title="$$\vec{x}^{(i)} = (1, x_1^{(i)}, x_2^{(i)}, \cdots, x_n^{(i)})^{T} \quad (i=1, 2, \cdots, m)" /></a>
>
> <!--X_{m \times n}=\left(\begin{matrix}1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\\vdots & \vdots & \vdots & \ddots & \vdots \\1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\ \end{matrix}\right)-->
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=X_{m&space;\times&space;n}=\left(\begin{matrix}1&space;&&space;x_1^{(1)}&space;&&space;x_2^{(1)}&space;&&space;\cdots&space;&&space;x_n^{(1)}&space;\\1&space;&&space;x_1^{(2)}&space;&&space;x_2^{(2)}&space;&&space;\cdots&space;&&space;x_n^{(2)}&space;\\\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\1&space;&&space;x_1^{(m)}&space;&&space;x_2^{(m)}&space;&&space;\cdots&space;&&space;x_n^{(m)}&space;\\&space;\end{matrix}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{m&space;\times&space;n}=\left(\begin{matrix}1&space;&&space;x_1^{(1)}&space;&&space;x_2^{(1)}&space;&&space;\cdots&space;&&space;x_n^{(1)}&space;\\1&space;&&space;x_1^{(2)}&space;&&space;x_2^{(2)}&space;&&space;\cdots&space;&&space;x_n^{(2)}&space;\\\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\1&space;&&space;x_1^{(m)}&space;&&space;x_2^{(m)}&space;&&space;\cdots&space;&&space;x_n^{(m)}&space;\\&space;\end{matrix}\right)" title="X_{m \times n}=\left(\begin{matrix}1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\\vdots & \vdots & \vdots & \ddots & \vdots \\1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\ \end{matrix}\right)" /></a>
>
>  
> Feature coefficiency:
>
> <!--$$\vec{\theta} = (\theta_0, \theta_1, \theta_2, \cdots, \theta_n)^T $$-->
>
><a href="https://www.codecogs.com/eqnedit.php?latex=$$\vec{\theta}&space;=&space;(\theta_0,&space;\theta_1,&space;\theta_2,&space;\cdots,&space;\theta_n)^T&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\vec{\theta}&space;=&space;(\theta_0,&space;\theta_1,&space;\theta_2,&space;\cdots,&space;\theta_n)^T&space;$$" title="$$\vec{\theta} = (\theta_0, \theta_1, \theta_2, \cdots, \theta_n)^T $$" /></a>
> 
> The linear regression model:
>
> <!--$$h_\theta(x^{(i)})= \theta_0\cdot 1 + \theta_1\cdot x_1^{(i)} + \theta_2\cdot x_2^{(i)} + \cdots + \theta_n\cdot x_n^{(i)}$$-->
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=$$h_\theta(x^{(i)})=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;x_1^{(i)}&space;&plus;&space;\theta_2\cdot&space;x_2^{(i)}&space;&plus;&space;\cdots&space;&plus;&space;\theta_n\cdot&space;x_n^{(i)}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$h_\theta(x^{(i)})=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;x_1^{(i)}&space;&plus;&space;\theta_2\cdot&space;x_2^{(i)}&space;&plus;&space;\cdots&space;&plus;&space;\theta_n\cdot&space;x_n^{(i)}$$" title="$$h_\theta(x^{(i)})= \theta_0\cdot 1 + \theta_1\cdot x_1^{(i)} + \theta_2\cdot x_2^{(i)} + \cdots + \theta_n\cdot x_n^{(i)}$$" /></a>
>
> Or
>
> <!--$$ h_\theta(X) = X\cdot\vec{\theta}$$-->
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;h_\theta(X)&space;=&space;X\cdot\vec{\theta}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;h_\theta(X)&space;=&space;X\cdot\vec{\theta}$$" title="$$ h_\theta(X) = X\cdot\vec{\theta}$$" /></a>


+ Models:
	- Discrete output: **Classification**;
	- Continuous output: **Regression**;


### 2. Cost Function
A good model will produce results h<sub>θ</sub> which has the least difference with the right answer Y, 

That is to minimize the cost function J(θ):

<!--$$J(\theta) = \sum^m_{i=1} (h_\theta(\vec{x}^{(i)})-Y^{(i)})^2$$-->

<a href="https://www.codecogs.com/eqnedit.php?latex=$$J(\theta)&space;=&space;\sum^m_{i=1}&space;(h_\theta(\vec{x}^{(i)})-Y^{(i)})^2$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$J(\theta)&space;=&space;\sum^m_{i=1}&space;(h_\theta(\vec{x}^{(i)})-Y^{(i)})^2$$" title="$$J(\theta) = \sum^m_{i=1} (h_\theta(\vec{x}^{(i)})-Y^{(i)})^2$$" /></a>

Below is one of the way to get optimal θ

### 3. Gradient Descent

This is to find θ, so that 

<!--\frac{\partial J(\theta)}{\partial \theta_j} = 0, \quad j = 1,2, ..., n-->

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;J(\theta)}{\partial&space;\theta_j}&space;=&space;0,&space;\quad&space;j&space;=&space;1,2,&space;...,&space;n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J(\theta)}{\partial&space;\theta_j}&space;=&space;0,&space;\quad&space;j&space;=&space;1,2,&space;...,&space;n" title="\frac{\partial J(\theta)}{\partial \theta_j} = 0, \quad j = 1,2, ..., n" /></a>

"Gradient descent" is to simutaneously change each θ<sub>j</sub>'s value along with their partial derivative, iterate until all partial derivative equals to zero (or a weaken condition, the cost function value less than a threshold).

<!--\theta_j' = \theta_j - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta_j}, \quad j = 1,2, ..., n-->

<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_j'&space;=&space;\theta_j&space;-&space;\alpha&space;\cdot&space;\frac{\partial&space;J(\theta)}{\partial&space;\theta_j},&space;\quad&space;j&space;=&space;1,2,&space;...,&space;n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_j'&space;=&space;\theta_j&space;-&space;\alpha&space;\cdot&space;\frac{\partial&space;J(\theta)}{\partial&space;\theta_j},&space;\quad&space;j&space;=&space;1,2,&space;...,&space;n" title="\theta_j' = \theta_j - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta_j}, \quad j = 1,2, ..., n" /></a>

In the above equation, __"α"__ determines step length (constant), __"partial derivative"__ determine both step length (decrease when proning to 0) and the orientation (increase to 0 or decrease to 0).
For a large positive partial derivative, θ will decrease fast (imaging a car sliding fast at a steep slope); and for a small negative partial derivative, θ will increase slow (imaging a car sliding fast at a steep slope), they are all inclined to zero. **This could explain only one constant α is required in the equation.**

<div align=center>
	<img width="500" height="285" src="https://github.com/CnDE-M/Coursera_MarchineLearning/blob/master/Week_1_Gradient_Descent/svgs/gradient_descent.png"/>
</div>

Let's simplify the equation:
<!--$$\frac{\partial{J(\vec{\theta})}}{\partial{\theta_j}} = 2 \times \theta_j \times \sum^m_{i=1} (h_\theta(\vec{x}^{(i)})-Y^{(i)}), \quad j=1, 2, \cdots. n $$-->

<a href="https://www.codecogs.com/eqnedit.php?latex=$$\frac{\partial{J(\vec{\theta})}}{\partial{\theta_j}}&space;=&space;2&space;\times&space;\theta_j&space;\times&space;\sum^m_{i=1}&space;(h_\theta(\vec{x}^{(i)})-Y^{(i)}),&space;\quad&space;j=1,&space;2,&space;\cdots.&space;n&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\frac{\partial{J(\vec{\theta})}}{\partial{\theta_j}}&space;=&space;2&space;\times&space;\theta_j&space;\times&space;\sum^m_{i=1}&space;(h_\theta(\vec{x}^{(i)})-Y^{(i)}),&space;\quad&space;j=1,&space;2,&space;\cdots.&space;n&space;$$" title="$$\frac{\partial{J(\vec{\theta})}}{\partial{\theta_j}} = 2 \times \theta_j \times \sum^m_{i=1} (h_\theta(\vec{x}^{(i)})-Y^{(i)}), \quad j=1, 2, \cdots. n $$" /></a>

To formalize the simplyify the equation, 1/2m is multipled in the front, and the θ gradient descent is:

<!--$$\theta_j' = \theta_j - \alpha \times \frac{\theta_j}{m} \times \sum^m_{i=1} (h_\theta(\vec{x}^{(i)})-Y^{(i)}), \quad j=1, 2, \cdots. n $$-->

<a href="https://www.codecogs.com/eqnedit.php?latex=$$\theta_j'&space;=&space;\theta_j&space;-&space;\alpha&space;\times&space;\frac{\theta_j}{m}&space;\times&space;\sum^m_{i=1}&space;(h_\theta(\vec{x}^{(i)})-Y^{(i)}),&space;\quad&space;j=1,&space;2,&space;\cdots.&space;n&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\theta_j'&space;=&space;\theta_j&space;-&space;\alpha&space;\times&space;\frac{\theta_j}{m}&space;\times&space;\sum^m_{i=1}&space;(h_\theta(\vec{x}^{(i)})-Y^{(i)}),&space;\quad&space;j=1,&space;2,&space;\cdots.&space;n&space;$$" title="$$\theta_j' = \theta_j - \alpha \times \frac{\theta_j}{m} \times \sum^m_{i=1} (h_\theta(\vec{x}^{(i)})-Y^{(i)}), \quad j=1, 2, \cdots. n $$" /></a>

### 4. Tips

1. Prepossess data

+ feature scaling

<!--$$x_i' = \frac{x_i}{x_{max}-x_{min}}$$-->

<a href="https://www.codecogs.com/eqnedit.php?latex=x_i'&space;=&space;\frac{x_i}{x_{max}-x_{min}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i'&space;=&space;\frac{x_i}{x_{max}-x_{min}}" title="x_i' = \frac{x_i}{x_{max}-x_{min}}" /></a>

+ mean normalization

<!--$$ x' = \frac{x - \mu}{\delta} $$-->

<a href="https://www.codecogs.com/eqnedit.php?latex=x'&space;=&space;\frac{x&space;-&space;\mu}{\delta}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x'&space;=&space;\frac{x&space;-&space;\mu}{\delta}" title="x' = \frac{x - \mu}{\delta}" /></a>

2. plot out J(θ)~iteration time
Check if J(θ) decrease with the increase of iteration time, this will lead you find a proper α。

<div align=center>
	<img width="500" height="285" src="https://github.com/CnDE-M/Coursera_MarchineLearning/blob/master/Week_1_Gradient_Descent/svgs/"/>
</div>

3. Polymonial regression model
Simply treat polymonial vaiable as a new feature.

<!--$$ h_\theta(x)= \theta_0\cdot 1 + \theta_1\cdot x + \theta_2\cdot x^2 + \theta_3\cdot x^3  $$-->

<a href="https://www.codecogs.com/eqnedit.php?latex=h_\theta(x)=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;x&space;&plus;&space;\theta_2\cdot&space;x^2&space;&plus;&space;\theta_3\cdot&space;x^3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_\theta(x)=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;x&space;&plus;&space;\theta_2\cdot&space;x^2&space;&plus;&space;\theta_3\cdot&space;x^3" title="h_\theta(x)= \theta_0\cdot 1 + \theta_1\cdot x + \theta_2\cdot x^2 + \theta_3\cdot x^3" /></a>

equals to:
<!--$$ h_\theta(t)= \theta_0\cdot 1 + \theta_1\cdot t_1 + \theta_2\cdot t_2 + \theta_3\cdot t_3, \quad$$ \begin{cases} t_1 = x \\t_2 = x^2\\t_3 = x^3\end{cases}-->

<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;h_\theta(t)=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;t_1&space;&plus;&space;\theta_2\cdot&space;t_2&space;&plus;&space;\theta_3\cdot&space;t_3,&space;\quad$$&space;\begin{cases}&space;t_1&space;=&space;x&space;\\&space;t_2&space;=&space;x^2\\&space;t_3&space;=&space;x^3&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;h_\theta(t)=&space;\theta_0\cdot&space;1&space;&plus;&space;\theta_1\cdot&space;t_1&space;&plus;&space;\theta_2\cdot&space;t_2&space;&plus;&space;\theta_3\cdot&space;t_3,&space;\quad$$&space;\begin{cases}&space;t_1&space;=&space;x&space;\\&space;t_2&space;=&space;x^2\\&space;t_3&space;=&space;x^3&space;\end{cases}" title="$$ h_\theta(t)= \theta_0\cdot 1 + \theta_1\cdot t_1 + \theta_2\cdot t_2 + \theta_3\cdot t_3, \quad$$ \begin{cases} t_1 = x \\ t_2 = x^2\\ t_3 = x^3 \end{cases}" /></a>

