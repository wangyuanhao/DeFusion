function [L, LEhF, LEh, outL] = snfCal(D, K, neBool)
alpha = 0.4;
T = 20;
if nargin < 3
    neBool = true;
end

dataNum = length(D);
indW = cell(1, dataNum);
indWEh = cell(1, dataNum);
for i = 1:dataNum
    dataScore = D{i};
    Data = Standard_Normalization(dataScore);
    Dist = dist2(Data, Data);
    indW{i} = affinityMatrix(Dist, K, alpha);
    indWEh{i} = Network_Enhancement(indW{i}, 2, K, 0.7);
end

W = SNF(indW, K, T);
WEhF = Network_Enhancement(W, 2, K, 0.7);
WEh = SNF(indWEh, K, T);
% WEhEh = Network_Enhancement(WEh, 2, 20, 0.7);
WEhEh = Network_Enhancement(WEh, 2, K, 0.7);


D = diag(sum(W, 2));
L = D - W;

DEh = diag(sum(WEh, 2));
LEh = DEh - WEh;


DEhF = diag(sum(WEhF, 2));
LEhF = DEhF - WEhF;

DEhEh = diag(sum(WEhEh, 2));
if neBool
    outL = DEhEh - WEhEh;
else
    outL = L;
end

end