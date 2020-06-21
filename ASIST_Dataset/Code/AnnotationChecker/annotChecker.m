% annotChecker.m
%
% Kemal Tugrul Yesilbek
% 
% April 2014
%
% Visualizes sketches and annotations made on them.
%
%

%% Initialize 
close all;
clear all;
clc;

%% Let user select root path
rootFolder = uigetdir(pwd, 'Select the folder contains scenes');

%% Get files in folder
xmlFileNames = getFiles(rootFolder, true, '.xml');			% Get .xml files
annotFileNames = getFiles(rootFolder, false, 'annot_');	 % Get annotation files

%% Fix here -------------------------------------------
idx = 1;
for i = 1 : length(xmlFileNames)
	if(xmlFileNames{i}(1) ~= '.')
		newFileNames{idx} = xmlFileNames{i};
		idx = idx + 1;
	end
end
xmlFileNames = newFileNames;
%%-----------------------------------------------------------------

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


for f = 1 : length(xmlFileNames)

fprintf('%d of %d...\n', f, length(xmlFileNames));

close all;
	
%% Fire up selection window
%selectFileToAnnotate; % % The output will be in the base workspace as var. with name "selectedFileIdx"
selectedFileName = xmlFileNames{f};
trf = [rootFolder '\' selectedFileName];

%% Visualize
annotChecker_main(trf);

end











