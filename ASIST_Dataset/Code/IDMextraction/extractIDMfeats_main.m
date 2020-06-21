% extractIDMfeats_main.m
%
% Kemal Tugrul Yesilbek
% April 2015
%
% Extracts IDM features of the combinations
%

%% Initialize
close all;
clear all;
clc;

%% Options
combFile = 'combinations.mat';

%% Load data
load(combFile);

%% Extract IDM features
feats = ones(length(combinations), 720) * -1;
for c = 1 : length(combinations)
   fprintf('File: %d of %d...\n', c, length(combinations));
   %out{c} = extractIDMfeats(combinations{c});
   feats(c,:) = extractIDMfeats(combinations{c});
end

save('IDMfeats.mat', 'feats');






