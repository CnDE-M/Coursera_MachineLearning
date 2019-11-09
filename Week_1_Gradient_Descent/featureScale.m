function [new_x] = featureScale(old_x)
%% description:
% normalize x data by divide it by x range (x_max - x_min)
%% Input Args:
% "old_x": x data to be processed;
% if x is a matrix, normalize will performed by column
%% Output Args:
% "new_x": normalized x

for ii = 1: size(old_x,1)
    new_x(ii,:) = old_x(ii,:) ./ (max(old_x,[],1) - min(old_x,[],1));
end

end