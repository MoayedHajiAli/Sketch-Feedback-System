% fillPreprocessMain.m
%
% Kemal Tugrul Yesilbek
% June 2015
%
% Test bed for filled sketch detection and processing
%
%

%% Initialization
close all;
clear all;
clc;
fprintf('Script Start...\n');

%% Options
combData = 'C:\Users\kemal\Dropbox\Bin\Study\Research\1003\SoftWorks\PosInstExtraction\Utilities_SideWorks\FilledSketchPreprocess\Data\combinations.mat';
detectionRatioThreshold = 0.08;
detectionPointThreshold = 50;

%% Load data
fprintf('Loading Data...\n');
load(combData);

%% Detection
filledLog = ones(length(combinations), 1) * -1;
for i = 1 : length(combinations)
    fprintf('Detection -- File: %d of %d...\n', i, length(combinations));
    
    [filledLog(i), ratio] = isFilled( combinations{i}.coords, detectionRatioThreshold, detectionPointThreshold );
    
%     if(filledLog(i))
%         printCoords( combinations{i}.coords );
%         title( [num2str(filledLog(i)), ' -- ', num2str(ratio)] );
%         drawnow;
%         pause(1);
%     end
    
end
filledIdx = find(filledLog == 1);

%% Processing
for i = 1 : length(filledIdx)
    fprintf('Processing -- File: %d of %d...\n', i, length(filledIdx));
    
    idx = filledIdx(i);
    %combinations{idx}.coords = processFilled( combinations{idx}.coords );
    combinations{idx}.coords = unique(combinations{idx}.coords, 'rows');
    tmp = processFilled( combinations{idx}.coords );
    
    % Display Results
    printCoords(combinations{idx}.coords);
    printCoords(tmp);
    for i = 1:length(tmp)
       plot(tmp(i,1), tmp(i,2), 'ro'); 
       text(tmp(i,1)+1, tmp(i,2)+1, num2str(i));
    end
    
    drawnow;
    pause(1);
    
end


























