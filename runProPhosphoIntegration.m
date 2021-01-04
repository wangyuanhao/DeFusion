clear;clc;close all;

addpath(genpath('./npy-matlab/'));
addpath(genpath('./SNFmatlab/'))
addpath(genpath('./Network_Enhancement/'))
addpath(genpath('./DeFusion/'))

lowDim = 2;
alpha = 1;
gamma = 1;
K = 20;

dir = './data/Protein&Phosphoprotein/';
view = {'protein', 'phosphoprotein'};

protein = readNPY(strcat(dir, '/protein.npy'));
phosphoprotein = readNPY(strcat(dir, '/phosphoprotein.npy'));

dataCell = cell(length(view), 1);
dataCell{1} = protein;
dataCell{2} = phosphoprotein;

  
fout = sprintf('lowDim=%d_alpha=%.2f_gamma=%.3f.mat', lowDim, alpha, gamma);

outPath = strcat(dir, fout);
[X, Z, E, convergence] = DeFusion(dataCell, lowDim, alpha, gamma, K, outPath);

fileOutNameC = strcat('lowDim=', num2str(lowDim),'_alpha=', num2str(alpha), ...
    '_gamma=', num2str(gamma),'_X.csv');
fileOutPathC = strcat(dir, fileOutNameC);
dlmwrite(fileOutPathC, X);	

