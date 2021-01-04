function [X] = proxL1Solver(Y, alpha)
    [nRow, nCol] =  size(Y);
    X = zeros(nRow, nCol);
    X(Y > alpha) =  Y(Y>alpha) -  alpha;
    X(Y < - alpha) = Y(Y<-alpha) + alpha;
end