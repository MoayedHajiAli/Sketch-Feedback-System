% annot_v2_main.m
%
% Kemal Tugrul Yesilbek
% April 2015
%
%

%% Initialize
close all;
clear all;
clc;


%% Let user select root path
rootFolder = uigetdir(pwd, 'Select the folder contains scenes');

while(true)
%% Get files in folder
xmlFileNames = getFiles(rootFolder, true, '.xml');			% Get .xml files
annotFileNames = getFiles(rootFolder, false, 'annot_');	 % Get annotation files


%% Get annot status of files
% Get annotation file's annotated file names
annotName = {};
for i = 1 : length(annotFileNames)
	annotName{i} = extractAnnotFileName( annotFileNames{i} );
end

if( ~isempty(annotName) )
	annotName = unique(annotName); % Remove redundant entries
end

% Decide annotation status
annotStat = zeros( length(xmlFileNames), 1 );
for i = 1 : length(annotStat)
	indexC = strfind(annotName, xmlFileNames{i});
	index = find(not(cellfun('isempty', indexC)));
	
	if( ~isempty(index) )
		annotStat(i) = 1;
	end
end

%% Label annotated file names
global labeledNames;
labeledNames = xmlFileNames;
for i  = 1 : length(xmlFileNames)
	if(annotStat(i) == 1)
		labeledNames{i} = ['(OK) ' xmlFileNames{i}];
	end
end


%% Fire up selection window
selectFileToAnnotate; % % The output will be in the base workspace as var. with name "selectedFileIdx"
selectedFileName = xmlFileNames{selectedFileIdx};
trf = [rootFolder '\' selectedFileName];

%% Annotate

annot_v2(trf);

%% Restart
end





