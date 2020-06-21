% visualizeCombinations.m
%
% Kemal Tugrul Yesilbek
% May 2015
%
% Visualizes Combinations
%

%% Initialization
close all;
clear all;
clc;

%% Options
labelToVis = -1;

%% Load data
labelDir    = 'C:\Users\kemal\Dropbox\Bin\Study\Research\1003\SoftWorks\SketchDatabases\BalanceQuestion\labels.mat';
combsDir    = 'C:\Users\kemal\Dropbox\Bin\Study\Research\1003\SoftWorks\SketchDatabases\BalanceQuestion\combinations.mat';
outDir      = 'C:\Users\kemal\Dropbox\Bin\Study\Research\1003\SoftWorks\DataPreperation\DataAnalysis\VisOut\'; 

tmp = load(combsDir);
combs = tmp.combinations;

tmp = load(labelDir);
labels = tmp.labels;

%% Show Data with Label
idxToVis = find(labels == labelToVis);

for i = 1 : length(idxToVis)
    
    % Print combination
    fprintf('%d of %d...\n', i, length(idxToVis));
    plotComb( combs{ idxToVis(i) }, 'bo' );
    label = ['Label: ' num2str( labels( idxToVis(i) ))]; 
    shg;
    title(label);
    
    % Save to disk
    fname = [num2str(labels(idxToVis(i))), '_', num2str(i)];
    saveas(gcf, [outDir, fname, '.jpeg'], 'jpeg');
    close;
    
end



























