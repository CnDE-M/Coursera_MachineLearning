function [new_x] = meanNorm(old_x)
%% description:
% normalize x data by divide it by [x - mean(x)] / std(x)
%% Input Args:
% "old_x": x data to be processed;
%  if x is a matrix, normalize will performed by column
%% Output Args:
% "new_x": normalized x

col_stable = find( std(old_x, 1) ==0);
miu = mean(old_x, 1);
theta = std(old_x, 1);
miu(:,col_stable) = 0;
theta(:,col_stable) = 1;

for ii = 1: size(old_x,1)
    new_x(ii,:) = ( old_x(ii,:) - miu )./ theta;
end

end