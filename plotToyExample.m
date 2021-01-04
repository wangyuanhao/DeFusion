clc;clear;close all;
warning('off');
addpath(genpath('./subplot_tight/'))

low_dim = 3;
alpha = 0.1;
gamma = 10;
K = 20;
groundTruth = [ones(30,1);2*ones(30,1);3*ones(30,1)];

%% Load Data
tdir = './data/simulation_data/';
fi = sprintf('./data/simulation_data/toyexample.csv');
data = csvread(fi,1, 0);

outFi = sprintf('toyexample_alpha=%.2f_gamma=%.3f_K=%d.mat', alpha, gamma, K);
outPath = strcat(tdir, outFi);
load(outPath);

D = cell(4,1);
D{1} = data(:,1:120);
D{2} = data(:, 121:330);
D{3} = data(:, 331:1530);
D{4} = X_f;
%% Show heatmap of data matrices
showData(D)

%% Show latent sample representation
label = ones(size(data, 1), 1);
label(31:60) = 2;
label(61:90) = 3;
d = 2;

xLim = [-4, 8; -5, 10; -2, 30; -1, 2];
yLim = [-5, 10; -5, 10; -10, 20; -1.5, 1.5];

for i = 1:length(D)
    pcaDi = pcaTransform(D{i}, d);
    if i == 4
        lgd = true;
    else
        lgd = false;
    end
    plotScatter(pcaDi, label, lgd, xLim(i, :), yLim(i, :));
end

%% plot function
function [pcaX] = pcaTransform(X, d)
% centerX = X - mean(X, 1);
coeff = pca(X, 'Centered', true);
pcaX = X*coeff(:, 1:d);
end

function [] = plotScatter(data, label, lgd, xLim, yLim)
figure('Position', [400, 400, 300, 240])
uqLabel = unique(label);
shape = {'o', '+', 'x'};
color = [0.47, 0.67, 0.19; 0, 0.45, 0.74; 0.85, 0.33, 0.10];
% color = {'r', 'b', 'g'};
center = zeros(length(uqLabel), 2);
for i = 1:length(uqLabel)
    center(i, :) = mean(data(label==uqLabel(i), :), 1);
end
x1 = (min(xLim)+0.01):0.01:(max(xLim)-0.01); %0.1, 0.1, 0.2, 0.03
x2 = (min(yLim)+0.01):0.01:(max(yLim)-0.01);
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot

idx2Region = kmeans(XGrid,3,'MaxIter',1,'Start',center);
gscatter(XGrid(:,1),XGrid(:,2),idx2Region,...
    [0.92,0.92,0.92;0.96,0.96,0.96; 1.0, 1.0, 1.0],'..');
legend('off')
xlim(xLim)
ylim(yLim)
hold on
scatterHandle = ones(length(uqLabel), 1);
for i = 1:length(uqLabel)
    if shape{i} == 'x' || shape{i} == '+'
        p = scatter(data(label==uqLabel(i), 1), data(label==uqLabel(i), 2), 24, color(i, :), shape{i}, 'LineWidth', 1.5);
    else
        p = scatter(data(label==uqLabel(i), 1), data(label==uqLabel(i), 2), 18, color(i, :), shape{i}, 'LineWidth', 1.5);
    end
    scatterHandle(i) = p;
    hold on  
end
if lgd
    legend(scatterHandle, {'Group 1', 'Group 2', 'Group 3'})
end

set(gca, 'FontName', 'Times', 'FontSize', 10);
box on


end


function [] = showData(D)
% figure('Position', [587 399 360 306]);
figure('Position', [661 341 894 311]);
subplot_tight(1, 3, 1, [0.08, 0.04])
D1= zscore(D{1});
% D1(D1<median(D1(:))) = 0;
D1 = minMaxScalar(D1);
D1(bsxfun(@lt, D1, median(D1))) = 0;
imagesc(D1)
%title('view1', 'FontSize', 16, 'FontWeight', 'bold')
axis off
subplot_tight(1, 3, 2, [0.08, 0.04])

D2= zscore(D{2});
D2 = minMaxScalar(D2);
% D2(D2<median(D2(:))) = 0;
D2(bsxfun(@lt, D2, median(D2))) = 0;
imagesc(D2)
%title('view2', 'FontSize', 16, 'FontWeight', 'bold')
axis off
subplot_tight(1, 3, 3, [0.08, 0.04])

D3= zscore(D{3});
% D3(D3<median(D3(:))) = 0;
D3 = minMaxScalar(D3);
D3(bsxfun(@lt, D3, median(D3))) = 0;
imagesc(D3)
%title('view3', 'FontSize', 16, 'FontWeight', 'bold')
axis off
end

function [scaledD] = minMaxScalar(D)
 scaledD = bsxfun(@minus, D, min(D));
 scaledD = 10*bsxfun(@rdivide, scaledD, max(D)-min(D));
end
