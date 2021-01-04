function [X] = proxL21Solver(Y, alpha, dim)

if nargin < 3
    dim = 1;
end

[nRow, nCol] = size(Y);
X = zeros(nRow, nCol);
if dim == 1
    for j = 1:nCol
        if norm(Y(:, j), 'fro') >= alpha
            X(:, j) = (1 - alpha /norm(Y(:,j),'fro')) * Y(:,j);
        end
    end
else
    for i = 1:nRow
        if norm(Y(i,:),'fro') >= alpha
            X(i, :) = (1-alpha/nrom(Y(i, :))) * Y(i, :);
        end
    end
end