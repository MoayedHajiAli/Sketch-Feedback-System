% testdpfrag.m
%
%

%% Initialize
clear all;
close all;
clc;
warning off;

%% Load classifier
load env_nicicon;


%% Select and files
% Get xml
rootFolder = uigetdir(pwd, 'Select the folder contains scenes');
xmlFileNames = getFiles(rootFolder, true, '.xml');			% Get .xml files

% Get destination
destinationFolder = uigetdir(pwd, 'Select the output folder');

%% Visualize
parfor f = 1 : length(xmlFileNames)
	
	file = [rootFolder '\' xmlFileNames{f}];
	current = get_strokes( file );
	
	clc;
	fprintf('File: %s\n', xmlFileNames{f});
	
	fig = figure('visible','off');
	hold on;
	for strk = 1 : length(current)
		current_stroke = current(strk);
		current_stroke = douglas_peucker(current_stroke, params.douglasthresh);
		current_stroke = resample(current_stroke, params.resample_interval, params.anglemeasure);     
		current_stroke = fragstroke(current_stroke, prec, params);
		plot_stroke(current_stroke);
	end
	title(xmlFileNames{f});
	
	% Save to the disk
	img = [destinationFolder '\' xmlFileNames{f} '.png'];
	saveas(fig,img);
	close all;
	
end
