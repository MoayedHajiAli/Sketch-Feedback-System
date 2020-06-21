function [ outLabels, objCount ] = selectInstances( labels, fileIDs, labelToUse, numToSelect )
% Select instances a positive, and return new labels
% There are multiple steps to do this and it is not straightforward. What
% we want to achieve is to label one positive instance from a sketch and
% label the rest of the combinations for this sketch as garbage. While
% doing this, we will indicate that combinations that are part of another
% sketches are unknown addition to the positive instances other than
% selected for that selected sketch.


%% Select Files That has Pos Class Randomly
objCount = countSketchObject(labels, labelToUse, fileIDs);

% Find non-zero count scenes
nonZeroCountIdx = find(objCount > 0);

% Select randomly
selectedFiles = nonZeroCountIdx( randperm( length(nonZeroCountIdx), numToSelect ) );

labelsGen = zeros(numToSelect, length(labels));
for f = 1 : numToSelect
    % Generate
    labelsGen(f,:) = genLabelsForOne(labels, labelToUse, fileIDs, selectedFiles(f));
end

%% Merge labels
outLabels = zeros(length(labels), 1);

for l = 1 : length(labels)
    
    if( any(labelsGen(:,l) == -1) )
        outLabels(l) = -1;
    elseif( any(labelsGen(:,l) == labelToUse) )
        outLabels(l) = labelToUse;
    else
        outLabels(l) = 0;
    end
    
end


end


function [outLabels] = genLabelsForOne(labels, labelToUse, fileIDs, selectedFile)


%% Label combinations (4 steps)

% 1) Label all combinations as -1 (garbage)
outLabels = ones( length(labels), 1 ) * -1;

% 2) Label combinations 0 (unknown) extracted from files other than
% selected as annotated
unknownCombsIdx = find( fileIDs ~= selectedFile );
outLabels(unknownCombsIdx) = 0;

% 3) Select a positive object from selected file and make it known
% (classNo)
selectedFileIdx = find( fileIDs == selectedFile );
selectedFileLabels = labels( selectedFileIdx );
selectedPosLabelsIdx = find( selectedFileLabels == labelToUse );
randomlySelectedPosLabelIdx = randi( length(selectedPosLabelsIdx), 1);

outLabels( selectedFileIdx( selectedPosLabelsIdx( randomlySelectedPosLabelIdx) ) ) = labelToUse;

% 4) Label other positive combinations as 0 (unknown)
invisiblePosLabelsIdx = setdiff(selectedPosLabelsIdx, selectedPosLabelsIdx( randomlySelectedPosLabelIdx));
outLabels( selectedFileIdx( invisiblePosLabelsIdx ) ) = 0;

end

function [count] = countSketchObject(labels, labelToTest, fileIDs)

	uniqueFilesIDs = unique(fileIDs); % FileIDs are 1,2,...,n order
	count = zeros( length(uniqueFilesIDs), 1 );
	
	for l = 1 : length(labels)
		if(labels(l) == labelToTest)
			count( fileIDs(l) ) = count( fileIDs(l) ) + 1;
        end
    end
    
end













