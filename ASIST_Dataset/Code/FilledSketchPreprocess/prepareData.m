% prepareData.m
%
% Kemal Tugrul Yesilbek
% July 2015
%
% Detects filled sketches in data and preprocess it, and save them back.
%

%% Initialization
close all;
clear all;
clc;
fprintf('Script Start...\n');
addpath( genpath(pwd) );

%% Options
isRunningOnCluster = isunix;

if(isRunningOnCluster)
    parpool local;
    combData = '/mnt/kufs/scratch/kyesilbek/FilledSketchPreprocess/Data/combinations.mat';
    labelsData = '/mnt/kufs/scratch/kyesilbek/FilledSketchPreprocess/Data/labels.mat';
else
    combData = 'C:\Users\kemal\Dropbox\Bin\Study\Research\1003\SoftWorks\DataPreperation\FilledSketchPreprocess\Data\combinations.mat';
    labelsData = 'C:\Users\kemal\Dropbox\Bin\Study\Research\1003\SoftWorks\DataPreperation\FilledSketchPreprocess\Data\labels.mat';
end

detectionRatioThreshold = 0.08;
detectionPointThreshold = 50;

%% Load data
fprintf('Loading Data...\n');
load(combData);
load(labelsData);

%% Detection
filledLog = ones(length(combinations), 1) * -1;
for i = 1 : length(combinations)
    fprintf('Detection -- File: %d of %d...\n', i, length(combinations));
    
    [filledLog(i), ratio] = isFilled( combinations{i}.coords, detectionRatioThreshold, detectionPointThreshold );
    
end
filledIdx = find(filledLog == 1);

%% Processing
for i = 1 : length(filledIdx)
    fprintf('Processing -- File: %d of %d...\n', i, length(filledIdx));
    idx = filledIdx(i);
    
    combinations{idx}.coords = unique(combinations{idx}.coords, 'rows');
    combinations{idx}.coords = processFilled( combinations{idx}.coords );
    combinations{idx}.strokeIDs = ones( length(combinations{idx}.coords), 1 );
    
end

%% Extract Features
% Make sure you add the apporipriate functions to Matlab path
fprintf('Extracting Features...\n');
feats = ones( length(combinations), 720 );
parfor i = 1 : length(combinations)
    feats(i,:) = extractIDMfeats( combinations{i} );
end

%% Save features
save('feats.mat', 'feats');
fprintf('Script End...\n');


    





















