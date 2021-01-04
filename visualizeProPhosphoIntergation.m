clc;clear;close all


dir = './data/Protein&Phosphoprotein/';
lowDim = 2;
alpha = 1;
gamma = 1;

labelFi = strcat(dir, 'proc_sample_label.csv');
labelDf = readtable(labelFi);
label = labelDf.label;


dataFi = strcat('lowDim=', num2str(lowDim),'_alpha=', num2str(alpha), ...
    '_gamma=', num2str(gamma),'_X.csv');
dataPath = strcat(dir, dataFi);

data = csvread(dataPath);

plotFigure(data, label)
acc = evalClustering(data, label);
display(acc);


function [] = plotFigure(data, label)
figure('Position', [257, 422, 420, 269])
subplot('Position', [0.1, 0.2, 0.5, 0.7])
scatter(data(label==0, 1), data(label==0, 2), 36, 'x')
hold on
scatter(data(label==1, 1), data(label==1, 2), 36, 'o')
xlabel('Dim 1')
ylabel('Dim 2')
xlim([-1, 4])
ylim([-1, 4])
% set(gca, 'YTickLabel', {'0', '', '1', '', '2', '', '3', ''})
set(gca, 'FontName', 'Times', 'FontSize', 10)
box on
lhd = legend({'N', 'T'});
% legend('Orientation','horizontal')
% pp = axes('position', [1, 0.6, 0.3, 0.3]);
subplot('Position', [0.7, 0.2, 0.15, 0.7])
[s, h] = silhouette(data, label);
set(gca, 'YTickLabel', {'N', 'T'}, 'FontName', 'Times', 'FontSize', 10)
display(mean(s))
end

function [acc] = evalClustering(data, y)
numSample = size(data, 1);
yPred = kmeans(data, 2, 'Replicates', 10) - 1;
acc = max(sum(y == yPred)/numSample, sum(y==(1-yPred)) / numSample);
end