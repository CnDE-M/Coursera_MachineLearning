# Script and Practice

In the file, several scripts are wrote to realize "gradient descent" method and generate a linear regression model with data from Machine Learning Repository<sup>[1]</sup> the "auto-mpg" dataset.

## auto_mpg dataset
Data from:
Asuncion, A. & Newman, D.J. (2007)<sup>[1]</sup>

Number of Instances: 398 (1st is deleted)
Number of Attributes: 8 (car name is deleted)
Attribute Information:
1. mpg:           continuous
2. cylinders:     multi-valued discrete
3. displacement:  continuous
4. horsepower:    continuous
5. weight:        continuous
6. acceleration:  continuous
7. model year:    multi-valued discrete
8. origin:        multi-valued discrete

## "gradientDescent.m"

  Function. 
  Realize gradient descent and return optimal θ, with J(θ)~iteration_time plot for checking if α is proper.

### Input Arguments:
1. dataset

<!--dataset = \left(\begin{matrix}y^{(1)} & 1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\ y^{(2)} &1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ y^{(m)} &1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\ \end{matrix}\right)-->

<a href="https://www.codecogs.com/eqnedit.php?latex=dataset&space;=&space;\left(&space;\begin{matrix}&space;y^{(1)}&space;&&space;1&space;&&space;x_1^{(1)}&space;&&space;x_2^{(1)}&space;&&space;\cdots&space;&&space;x_n^{(1)}&space;\\&space;y^{(2)}&space;&1&space;&&space;x_1^{(2)}&space;&&space;x_2^{(2)}&space;&&space;\cdots&space;&&space;x_n^{(2)}&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;y^{(m)}&space;&1&space;&&space;x_1^{(m)}&space;&&space;x_2^{(m)}&space;&&space;\cdots&space;&&space;x_n^{(m)}&space;\\&space;\end{matrix}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?dataset&space;=&space;\left(&space;\begin{matrix}&space;y^{(1)}&space;&&space;1&space;&&space;x_1^{(1)}&space;&&space;x_2^{(1)}&space;&&space;\cdots&space;&&space;x_n^{(1)}&space;\\&space;y^{(2)}&space;&1&space;&&space;x_1^{(2)}&space;&&space;x_2^{(2)}&space;&&space;\cdots&space;&&space;x_n^{(2)}&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;y^{(m)}&space;&1&space;&&space;x_1^{(m)}&space;&&space;x_2^{(m)}&space;&&space;\cdots&space;&&space;x_n^{(m)}&space;\\&space;\end{matrix}\right)" title="dataset = \left( \begin{matrix} y^{(1)} & 1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\ y^{(2)} &1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ y^{(m)} &1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\ \end{matrix}\right)" /></a>

2. pre_theta
<!--pre\_theta = \vec{\theta}^1= \begin{pmatrix} -1 \\ \theta_0^1 \\ \theta_1^1 \\ \vdots \\ \theta_n^1 \end{pmatrix}-->

<a href="https://www.codecogs.com/eqnedit.php?latex=pre\_theta&space;=&space;\vec{\theta}^1=&space;\begin{pmatrix}&space;-1&space;\\&space;\theta_0^1&space;\\&space;\theta_1^1&space;\\&space;\vdots&space;\\&space;\theta_n^1&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?pre\_theta&space;=&space;\vec{\theta}^1=&space;\begin{pmatrix}&space;-1&space;\\&space;\theta_0^1&space;\\&space;\theta_1^1&space;\\&space;\vdots&space;\\&space;\theta_n^1&space;\end{pmatrix}" title="pre\_theta = \vec{\theta}^1= \begin{pmatrix} -1 \\ \theta_0^1 \\ \theta_1^1 \\ \vdots \\ \theta_n^1 \end{pmatrix}" /></a>

3. alpha

1×1 double. Gradient step length;

4. itera_num

1×1 integer. Max iteration time，if cost function doesn't decrease below threshold;

5. theshold
1×1 double. A weak condition for optimal theta, that simply let cost function's value than a small value;


### Output Arguments

1. "opt_theta"

the optimal θ

2. "fg"

J(θ)~iteration time plot result:

<div align=center>
	<img width="500" height="285" src="https://github.com/CnDE-M/Coursera_MarchineLearning/blob/master/Week_1_Gradient_Descent/svgs/gradient_descent.png"/>
</div>


### Method

1. θ update
For the ii<sup>th</sup> iteration:

<!--
\vec{\theta}^{ii}_{2-n} = 
\begin{pmatrix}  \theta^{ii}_1 \\ \vdots \\ \theta^{ii}_n\end{pmatrix} = 
\begin{pmatrix}  \theta^{(ii-1)}_1 \\ \vdots \\ \theta^{(ii-1)}_n\end{pmatrix} - \frac{\alpha}{m} \times
\begin{pmatrix} x^{(1)}_1 & x^{(2)}_1 & \cdots & x^{(m)}_1 \\
x^{(1)}_2 & x^{(2)}_2 & \cdots & x^{(m)}_2 \\
\vdots & \vdots & \cdots & \vdots\\
x^{(1)}_n & x^{(2)}_n & \cdots & x^{(m)}_n \\
\end{pmatrix}_{n\times m} \times \quad
\left ( 
\left(\begin{matrix}y^{(1)} & 1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\ y^{(2)} &1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ y^{(m)} &1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\ \end{matrix}\right) \times 
\begin{pmatrix}  -1 \\ \theta^{ii}_0 \\ \theta^{ii}_1 \\ \vdots \\ \theta^{ii}_n\end{pmatrix}\right )-->

<a href="https://www.codecogs.com/eqnedit.php?latex=\vec{\theta}^{ii}_{2-n}&space;=&space;\begin{pmatrix}&space;\theta^{ii}_1&space;\\&space;\vdots&space;\\&space;\theta^{ii}_n\end{pmatrix}&space;=&space;\begin{pmatrix}&space;\theta^{(ii-1)}_1&space;\\&space;\vdots&space;\\&space;\theta^{(ii-1)}_n\end{pmatrix}&space;-&space;\frac{\alpha}{m}&space;\times&space;\begin{pmatrix}&space;x^{(1)}_1&space;&&space;x^{(2)}_1&space;&&space;\cdots&space;&&space;x^{(m)}_1&space;\\&space;x^{(1)}_2&space;&&space;x^{(2)}_2&space;&&space;\cdots&space;&&space;x^{(m)}_2&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\cdots&space;&&space;\vdots\\&space;x^{(1)}_n&space;&&space;x^{(2)}_n&space;&&space;\cdots&space;&&space;x^{(m)}_n&space;\\&space;\end{pmatrix}_{n\times&space;m}&space;\times&space;\quad&space;\left&space;(&space;\left(\begin{matrix}y^{(1)}&space;&&space;1&space;&&space;x_1^{(1)}&space;&&space;x_2^{(1)}&space;&&space;\cdots&space;&&space;x_n^{(1)}&space;\\&space;y^{(2)}&space;&1&space;&&space;x_1^{(2)}&space;&&space;x_2^{(2)}&space;&&space;\cdots&space;&&space;x_n^{(2)}&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;y^{(m)}&space;&1&space;&&space;x_1^{(m)}&space;&&space;x_2^{(m)}&space;&&space;\cdots&space;&&space;x_n^{(m)}&space;\\&space;\end{matrix}\right)&space;\times&space;\begin{pmatrix}&space;-1&space;\\&space;\theta^{ii}_0&space;\\&space;\theta^{ii}_1&space;\\&space;\vdots&space;\\&space;\theta^{ii}_n\end{pmatrix}\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\vec{\theta}^{ii}_{2-n}&space;=&space;\begin{pmatrix}&space;\theta^{ii}_1&space;\\&space;\vdots&space;\\&space;\theta^{ii}_n\end{pmatrix}&space;=&space;\begin{pmatrix}&space;\theta^{(ii-1)}_1&space;\\&space;\vdots&space;\\&space;\theta^{(ii-1)}_n\end{pmatrix}&space;-&space;\frac{\alpha}{m}&space;\times&space;\begin{pmatrix}&space;x^{(1)}_1&space;&&space;x^{(2)}_1&space;&&space;\cdots&space;&&space;x^{(m)}_1&space;\\&space;x^{(1)}_2&space;&&space;x^{(2)}_2&space;&&space;\cdots&space;&&space;x^{(m)}_2&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\cdots&space;&&space;\vdots\\&space;x^{(1)}_n&space;&&space;x^{(2)}_n&space;&&space;\cdots&space;&&space;x^{(m)}_n&space;\\&space;\end{pmatrix}_{n\times&space;m}&space;\times&space;\quad&space;\left&space;(&space;\left(\begin{matrix}y^{(1)}&space;&&space;1&space;&&space;x_1^{(1)}&space;&&space;x_2^{(1)}&space;&&space;\cdots&space;&&space;x_n^{(1)}&space;\\&space;y^{(2)}&space;&1&space;&&space;x_1^{(2)}&space;&&space;x_2^{(2)}&space;&&space;\cdots&space;&&space;x_n^{(2)}&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;y^{(m)}&space;&1&space;&&space;x_1^{(m)}&space;&&space;x_2^{(m)}&space;&&space;\cdots&space;&&space;x_n^{(m)}&space;\\&space;\end{matrix}\right)&space;\times&space;\begin{pmatrix}&space;-1&space;\\&space;\theta^{ii}_0&space;\\&space;\theta^{ii}_1&space;\\&space;\vdots&space;\\&space;\theta^{ii}_n\end{pmatrix}\right&space;)" title="\vec{\theta}^{ii}_{2-n} = \begin{pmatrix} \theta^{ii}_1 \\ \vdots \\ \theta^{ii}_n\end{pmatrix} = \begin{pmatrix} \theta^{(ii-1)}_1 \\ \vdots \\ \theta^{(ii-1)}_n\end{pmatrix} - \frac{\alpha}{m} \times \begin{pmatrix} x^{(1)}_1 & x^{(2)}_1 & \cdots & x^{(m)}_1 \\ x^{(1)}_2 & x^{(2)}_2 & \cdots & x^{(m)}_2 \\ \vdots & \vdots & \cdots & \vdots\\ x^{(1)}_n & x^{(2)}_n & \cdots & x^{(m)}_n \\ \end{pmatrix}_{n\times m} \times \quad \left ( \left(\begin{matrix}y^{(1)} & 1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\ y^{(2)} &1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ y^{(m)} &1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\ \end{matrix}\right) \times \begin{pmatrix} -1 \\ \theta^{ii}_0 \\ \theta^{ii}_1 \\ \vdots \\ \theta^{ii}_n\end{pmatrix}\right )" /></a>


and θ<sub>1-2</sub> keep as [-1, θ<sub>0</sub>] that they will not change value in iterations.

2. Cost function value

<!--J(\vec{\theta}^{ii}) = (1, 1, \cdots, 1)_{1\times m} \times \left ( \left(\begin{matrix}y^{(1)} & 1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\ y^{(2)} &1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ y^{(m)} &1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\ \end{matrix}\right) \times \begin{pmatrix}  -1 \\ \theta^{ii}_0 \\ theta^{ii}_1 \\ \vdots \\ \theta^{ii}_n\end{pmatrix}\right ) ^{.2}-->

<a href="https://www.codecogs.com/eqnedit.php?latex=J(\vec{\theta}^{ii})&space;=&space;(1,&space;1,&space;\cdots,&space;1)_{1\times&space;m}&space;\times&space;\left&space;(&space;\left(\begin{matrix}y^{(1)}&space;&&space;1&space;&&space;x_1^{(1)}&space;&&space;x_2^{(1)}&space;&&space;\cdots&space;&&space;x_n^{(1)}&space;\\&space;y^{(2)}&space;&1&space;&&space;x_1^{(2)}&space;&&space;x_2^{(2)}&space;&&space;\cdots&space;&&space;x_n^{(2)}&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;y^{(m)}&space;&1&space;&&space;x_1^{(m)}&space;&&space;x_2^{(m)}&space;&&space;\cdots&space;&&space;x_n^{(m)}&space;\\&space;\end{matrix}\right)&space;\times&space;\begin{pmatrix}&space;-1&space;\\&space;\theta^{ii}_0&space;\\&space;theta^{ii}_1&space;\\&space;\vdots&space;\\&space;\theta^{ii}_n\end{pmatrix}\right&space;)&space;^{.2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(\vec{\theta}^{ii})&space;=&space;(1,&space;1,&space;\cdots,&space;1)_{1\times&space;m}&space;\times&space;\left&space;(&space;\left(\begin{matrix}y^{(1)}&space;&&space;1&space;&&space;x_1^{(1)}&space;&&space;x_2^{(1)}&space;&&space;\cdots&space;&&space;x_n^{(1)}&space;\\&space;y^{(2)}&space;&1&space;&&space;x_1^{(2)}&space;&&space;x_2^{(2)}&space;&&space;\cdots&space;&&space;x_n^{(2)}&space;\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots&space;\\&space;y^{(m)}&space;&1&space;&&space;x_1^{(m)}&space;&&space;x_2^{(m)}&space;&&space;\cdots&space;&&space;x_n^{(m)}&space;\\&space;\end{matrix}\right)&space;\times&space;\begin{pmatrix}&space;-1&space;\\&space;\theta^{ii}_0&space;\\&space;theta^{ii}_1&space;\\&space;\vdots&space;\\&space;\theta^{ii}_n\end{pmatrix}\right&space;)&space;^{.2}" title="J(\vec{\theta}^{ii}) = (1, 1, \cdots, 1)_{1\times m} \times \left ( \left(\begin{matrix}y^{(1)} & 1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\ y^{(2)} &1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ y^{(m)} &1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \\ \end{matrix}\right) \times \begin{pmatrix} -1 \\ \theta^{ii}_0 \\ theta^{ii}_1 \\ \vdots \\ \theta^{ii}_n\end{pmatrix}\right ) ^{.2}" /></a>

And then plot out J(θ)~ iteration time.

### Example

Import data from Machine Learning Repository<sup>1</sup>. mpg is the dependent value, choose [cylinders, horsepower, weight, acceleration] to be independent value, or feature.

Set initial theta as [-1, 1, 1, 1, 1, 1], alpha = 0.01, iterate 20 times. 
theta = [-1.0000, 0.8262, 0.5533, 0.5952, 0.5226, 1.1135]


<div align=center>
	<img width="500" height="285" src="https://github.com/CnDE-M/Coursera_MachineLearning/blob/master/Week_1_Gradient_Descent/svgs/gradient_descent_1.png"/>
</div>


Then set initial theta as [-1, 0.8, 0.5, 0.5, 5, 1], alpha = 0.5, iterate 20 times. 
theta = [-1.0000, 0.0000, -0.8865, -0.9134, 0.6804, -0.3881]

<div align=center>
	<img width="500" height="285" src="https://github.com/CnDE-M/Coursera_MachineLearning/blob/master/Week_1_Gradient_Descent/svgs/gradient_descent_2.png"/>
</div>


Then set initial theta as [-1, 0.0, -0.88, -0.9, 0.6, -0.3], alpha = 0.5, iterate 100 times. 
theta = [-1.0000, -0.0000, -0.1037, -0.2592, -0.5235, -0.0272];

<div align=center>
	<img width="500" height="285" src="https://github.com/CnDE-M/Coursera_MachineLearning/blob/master/Week_1_Gradient_Descent/svgs/gradient_descent_3.png"/>
</div>

...
Until cost stable at around 114, theta = [-1.0000, -0.0000, -0.0860, -0.2240, -0.5656, -0.0105]

<div align=center>
	<img width="500" height="285" src="https://github.com/CnDE-M/Coursera_MachineLearning/blob/master/Week_1_Gradient_Descent/svgs/gradient_descent_4.png"/>
</div>


Let's check the result with Matlab function regression:
theta = [ -0.0000, -0.0863, -0.2240, -0.5653, -0.0106]
<div align=center>
	<img width="300" height="150" src="https://github.com/CnDE-M/Coursera_MachineLearning/blob/master/Week_1_Gradient_Descent/svgs/regress_test.PNG"/>
</div>


## Reference
[1] Asuncion, A. & Newman, D.J. (2007). UCI Machine Learning Repository [http://www.ics.uci.edu/~mlearn/MLRepository.html]. Irvine, CA: University of California, School of Information and Computer Science. 
