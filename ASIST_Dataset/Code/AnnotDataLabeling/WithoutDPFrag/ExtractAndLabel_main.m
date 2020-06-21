% ExtractAndLabel_main.m
%
% Extracts sketch piece combinations, compares all the combiantions with
% annotations and labels them
%
% April 2015
% Kemal Tugrul Yesilbek
%
% How to use:
%
% This program is to be used for extracting sketch piece combinations and
% compare those combinations with annotations made to label them. Labeling
% is done by point-by-point comparision of combinations and annotation
% sketch points. So, if only one point is different between a comb. and
% annot., comb. will not be recognized as annotated sketch piece.
%
% Store all of the sketches and annotations in seperate folders. When
% program is opened up, it will ask you to point those folders. Do not put
% annotations and sketches in sub-folders. Also, sketch files' names should
% be unique.
%
% Annotations should be produced by the sketch annotator. Other annotation
% structures will not work with this program.
%
% Labels available in annotations should be stored in \Labels\labels.mat
% file. As this program labels combinations with integers, labels file must
% not be changed. Program takes the place of a label string in labels file
% and assigns combinations with the index of that.
%
% The program will generate an error if an annotation is not found in
% combinations for a sketch. So, keep an eye on the command window to
% capture annotation mistakes and reannotate that sketch if neccessary.
%


%% Initialize
clear all;
close all;
clc;

%% Ask user to select folder where sketches are and get files
% Fire up a GUI
rootFolderXML = uigetdir(pwd, 'Select the folder contains sketches');
rootFolderAnnot = uigetdir(pwd, 'Select the folder contains annotations');

% Load files
xmlFileNames = getFiles(rootFolderXML, true, '.xml');			% Get .xml files
annotFileNames = getFiles(rootFolderAnnot, false, 'annot_');	 % Get annotation files
missingDestinationFolder = uigetdir(pwd, 'Select missing destination folder');
extraDestinationFolder = uigetdir(pwd, 'Select extra destination folder');


%% Start the main process

for file = 1 : length(xmlFileNames)
	fprintf('\nProcessing file %d of %d...\n', file, length(xmlFileNames));
	
	% Find corresponding annotation file names for file
	fprintf('Acquiring annotation files...\n');
	[ annotFileNamesForXML ] = getAnnotFileNames( xmlFileNames{file}, annotFileNames );
	
	% Add folder info to file names
	for fil = 1 : length(annotFileNamesForXML)
		annotFileNamesForXML{fil} = [rootFolderAnnot '\' annotFileNamesForXML{fil}];
	end
	
	xmlFile = [rootFolderXML '\' xmlFileNames{file}];
	
	% Label!
	[combinations{file}, labels{file}] = EandL( xmlFile, annotFileNamesForXML );
	
    % Find missing/extra piece
	[missing, extra] = findProblematicLabels(labels{file}, annotFileNamesForXML);
    
	if( ~isempty(missing) || ~isempty(extra) )
		% We have a problem Houston!
		fprintf('Some pieces are either missing or extra!\n');
		
		% Print
		if(~isempty(missing) && ~isempty(extra) )
			fprintf('Missing: %d \n Extra: %d\n', length( missing.count ), length( extra.count) );
		elseif( ~isempty(missing) )
			fprintf('Missing: %d\n', length( missing.count) );
		elseif( ~isempty(extra))
			fprintf('Extra: %d\n', length( extra.count) );
		end
		
		% Vis.
        if(~isempty(missing) )
            printProblematicSketch(xmlFile, xmlFileNames{file}, combinations{file}, labels{file}, missingDestinationFolder);
        else
            printProblematicSketch(xmlFile, xmlFileNames{file}, combinations{file}, labels{file}, extraDestinationFolder);
        end
	end
	
end

%% Save combinations and labels
fprintf('Saving data to disk...\n');

% Parse combinations var. so that each combination will be a seperate cell
combsParsed = {}; combsIndex = 1;
labelsParsed = [];

for sk = 1 : length(combinations)
	for comb = 1 : length(combinations{sk})
		combsParsed{combsIndex} = combinations{sk}{comb};
		labelsParsed(combsIndex) = labels{sk}(comb);
		correspondingFiles{combsIndex} = xmlFileNames{sk};
		
		combsIndex = combsIndex + 1;
	end
end

combinations = combsParsed;
labels = labelsParsed;

save('combinations.mat', 'combinations');
save('fileNames.mat', 'correspondingFiles');
save('labels.mat', 'labels');

%% End
fprintf('Script End...\n');































