% extractZernikefeats_main.m
%
% Kemal Tugrul Yesilbek
% April 2015
%
% Extracts Zernike features of the combinations
%

%% Initialize
close all;
clear all;
clc;

%% Options
% combFile = 'C:\Users\kemal\Dropbox\Bin\Study\Research\1003\SoftWorks\SketchDatabases\BalanceQuestion\combinations.mat';
% labelsFile = 'C:\Users\kemal\Dropbox\Bin\Study\Research\1003\SoftWorks\SketchDatabases\BalanceQuestion\labels.mat';
% fileNamesFile = 'C:\Users\kemal\Dropbox\Bin\Study\Research\1003\SoftWorks\SketchDatabases\BalanceQuestion\fileNames.mat';

combFile        = 'C:\Users\kemal\Google Drive\Bin\Study\Research\1003\SoftWorks\SketchDatabases\MatlabPrepared\1003\SerialResistorQuestion\combinations.mat';
labelsFile      = 'C:\Users\kemal\Google Drive\Bin\Study\Research\1003\SoftWorks\SketchDatabases\MatlabPrepared\1003\SerialResistorQuestion\labels.mat';
fileNamesFile   = 'C:\Users\kemal\Google Drive\Bin\Study\Research\1003\SoftWorks\SketchDatabases\MatlabPrepared\1003\SerialResistorQuestion\fileNames.mat';


%% Load data
load(combFile);
load(labelsFile);
load(fileNamesFile);

%% Extract IDM features
feats = ones( length(combinations), 70 ) * -1;
for c = 1 : length(combinations)
   fprintf('File: %d of %d...\n', c, length(combinations));
   %out{c} = extractZernike(combinations{c});
   feats = [feats;extractZernikefeats(combinations{c})];
end



