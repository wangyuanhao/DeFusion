clear;clc;close all;

addpath(genpath('./npy-matlab/'));
addpath(genpath('./SNFmatlab/'))
addpath(genpath('./Network_Enhancement/'))
addpath(genpath('./DeFusion/'))

lowDim = 6;
alpha = 1;
gamma = 10;
K = 20;

dir = './data/TCGA_LIHC/';
view = {'fpkm', 'mirnas', 'methy'};

fpkm = readNPY(strcat(dir, '/fpkm.npy'));
mirnas = readNPY(strcat(dir, '/mirnas.npy'));
methy = readNPY(strcat(dir, '/methy.npy'));

dataCell = cell(length(view), 1);
dataCell{1} = fpkm;
dataCell{2} = mirnas;
dataCell{3} = methy;

fout = sprintf('lowDim=%d_alpha=%.2f_gamma=%.3f.mat', lowDim, alpha, gamma);

outPath = strcat(dir, fout);
[X, Z, E, convergence] = DeFusion(dataCell, lowDim, alpha, gamma, K, outPath);

fileOutNameC = strcat('lowDim=', num2str(lowDim),'_alpha=', num2str(alpha), ...
    '_gamma=', num2str(gamma),'_X.csv');
fileOutPathC = strcat(dir, fileOutNameC);
dlmwrite(fileOutPathC, X);	