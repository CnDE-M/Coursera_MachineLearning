%% //////////////// DATA Library ///////////////// 
% ----------------------------------------------------------------------=-
%
% Data from:
% Asuncion, A. & Newman, D.J. (2007). UCI Machine Learning Repository 
% [http://www.ics.uci.edu/~mlearn/MLRepository.html]. Irvine, 
% CA: University of California, School of Information and Computer Science.
%
% Number of Instances: 398 (1st is deleted)
% Number of Attributes: 8 (car name is deleted)
% Attribute Information:
%     1. mpg:           continuous
%     2. cylinders:     multi-valued discrete
%     3. displacement:  continuous
%     4. horsepower:    continuous
%     5. weight:        continuous
%     6. acceleration:  continuous
%     7. model year:    multi-valued discrete
%     8. origin:        multi-valued discrete

    
clear
close all

dataURL = "http://mlr.cs.umass.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data";
dataset = webread(dataURL);
dataset = string(dataset);
dataset = split(dataset,'"');
dataset([1,2,797],:)=[];
dataset(2:2:794,:)=[];
dataset = split(dataset);
dataset(:,1)=[];
dataset = double(dataset);
dataset(:,9)=[];

% delete record with Nan
[row,col] = find( isnan(dataset)==1);
dataset(row,:)=[];

%% data and feature to use
mpg = dataset(:,1);
cylinders = dataset(:,2);
displacement = dataset(:,3);
horsepower = dataset(:,4);
weight = dataset(:,5);
acceleration = dataset(:,6);

%% //////////////// WEEK 1 ////////////////
%
%   Content:
%       COST FUNCTION & GRADIENT DESCENT
%
%  ////////////////////////////////////////////

%% Construct DataSet
trainDataset_y = mpg; % right answer
trainDataset_x = [cylinders, horsepower, weight, acceleration]; % features

% pre-process by mean normalization
trainDataset_y = meanNorm(trainDataset_y);
trainDataset_x = meanNorm(trainDataset_x);

trainDataset = [trainDataset_y, repmat(1,[size(trainDataset_y,1),1]),trainDataset_x];

%% model initial setting
ini_theta = [-1, 0.0, -0.08, -0.22, -0.56, -0.01];
alpha = 0.1;
itera_num = 100;
threshold = 1;

[theta, fg] = gradientDescent(trainDataset, ini_theta, alpha,  itera_num, threshold);


