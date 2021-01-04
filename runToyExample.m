clc;clear;close all

%% Add dependecies
addpath(genpath('./DeFusion/'))
addpath(genpath('./SNFmatlab/'))
addpath(genpath('./Network_Enhancement/'))

%% Parameters 
lowDim = 3;
alpha = 0.1;
gamma = 10;
K = 20;

%% Load data
tdir = './data/simulation_data/';
fi = sprintf('./data/simulation_data/toyexample.csv');
data = csvread(fi, 1, 0);

D = cell(3,1);
D{1} = data(:,1:120);
D{2} = data(:, 121:330);
D{3} = data(:, 331:1530); 

%% Run DeFusion      
outFi = sprintf('toyexample_alpha=%.2f_gamma=%.3f_K=%d.mat', alpha, gamma, K);
outPath = strcat(tdir, outFi);

[X, Z, E, convergence] = DeFusion(D, lowDim, alpha, gamma, K, outPath);

%% Save X                                
fileOutNameC = strcat('toyexample_alpha=', num2str(alpha), '_gamma=', ...
    num2str(gamma), '_K=', num2str(K),'_X.csv');
fileOutPathC = strcat(tdir,fileOutNameC);
dlmwrite(fileOutPathC, X);	
