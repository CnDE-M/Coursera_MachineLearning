function [opt_theta, fg] = gradientDescent(dataset, ini_theta, alpha,  itera_num, threshold)
%% description:
% return optimal thetas by gradient descent
%
%% Input Args:
% "dataset" = [Y, 1, X_1, X_2, ..., X_n];
% "pre_theta" = [-1, 1, theta_1, theta_2, ,,,, theta_n];
%       initial theta vector;
% alpha: gradient step length;
% "itera_num":
%       max iteration time if cost function doesn't decrease below
%       threshold;
% "threshold":
%       a weak condition for optimal theta, that simply let cost
%       function's value than a small value;
%
%% Output Args:
% "pot_theta": the optimal thetas
% "fg": J(theta)~iteration time plot result
%
m_sample = size(dataset,1); % feature number (from theta_0)
n_feature = size(dataset,2); % sample number
theta(:,1) = ini_theta;

for ii = 1:itera_num
    
    theta(:,ii+1) =[ini_theta(1)'; theta(2:n_feature,ii) - (alpha / m_sample) .* dataset(:,2:n_feature)' * (dataset * theta(:,ii))];
    
    % cost function
    J_cost(ii) = repmat(1,[1,m_sample]) *  (dataset * theta(:,ii+1)).^2;
    
    if J_cost(ii)< threshold
        J_cost(ii)
        break;
    end
    
end

opt_theta = theta(:,ii);

fg = figure();
plot([1:itera_num],J_cost,"b.-","lineWidth",2);
t = "¦Á= "+string(alpha)+"   iteration time = " + string(itera_num)+"  threshold = "+string(threshold);
title(t);
xlabel("iteration times");
ylabel("cost");
xlim([0,ii]);
xticks([0:ii]);
xticklabels(["",string([1:ii])]);
% dataset=[1,1,3,4;1,1,5,4;4,1,5,3;6,1,7,8;2,1,3,5;7,1,8,5];

end