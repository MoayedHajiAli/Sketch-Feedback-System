% convertFileNames.m
%
% Kemal Tugrul Yesilbek
% April 2015
%
% Convert file names associated with combinations to fileIDs, so it is
% easier to process
%

%% Initialize
close all;
clear all;
clc;
fprintf('Script Start...\n');

%% Load data
filePath = 'fileNames.mat';
outPath = 'fileIDs.mat';

load(filePath); % Returns "correspondingFiles" array of cell

%% Convert
% Get unique file names
uniqueFileNames = unique( correspondingFiles );

% Compare file names
fileIdx = ones( length(correspondingFiles), 1 ) * -1;

for f = 1 : length(correspondingFiles)
	for u = 1 : length(uniqueFileNames)
		
		if( strcmp( correspondingFiles{f}, uniqueFileNames{u} ) )
			fileIdx(f) = u;
		end
		
	end
end

%% Save output
save(outPath, 'fileIdx');
fprintf('Script End...\n');

