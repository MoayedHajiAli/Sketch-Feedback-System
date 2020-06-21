function [combinations, labels] =  EandL(  xmlFile, annotFiles )

%% Options
% This param. controls the #primitive lower-limit. If this is 2, then the
% combinations contains only 1 primitive wont be in the result. If this is
% n, then the combinations contanins 1:n-1 primitives wont be in the
% result.
combinationStart = 1;

% This param. controls the #primitive upper-limit. If this is 5 then the
% combinations contains more then 5 wont be in the result. If this is n,
% then the combinations contains n+1:Inf primitives wont be in the result.
combinationEnd = 15;

%% Load files
% Load sketch file
strokes = read_sketch(xmlFile);

% Load annotation files
annotStructs = loadAnnotFiles(annotFiles);

%% Fragment sketch

% Peek into annotations to see the threshold value (annotator adjusted)
if( isempty( annotStructs )  )
	DPthreshold = 10; % Just making up
else
    if(isempty(annotStructs{1}.DPthreshold))
        DPthreshold = 10; % Default setting
    else
        DPthreshold = annotStructs{1}.DPthreshold;    
    end
end

[ fragmented ] = fragmentFile( strokes, DPthreshold );

%% Extract Combinations
[combinations] = extractCombinations(fragmented, combinationStart, combinationEnd);



%% Find annotations and label them
if( isempty( annotStructs )  )
	labels = ones( length(combinations), 1) * -1;
	
else
	fprintf('Comparing combinations with annotations...\n');

	labels = ones( length(combinations), 1) * -1;
	
	for comb = 1 : length(combinations)
		for annot = 1 : length(annotStructs)
			
			labelString = compareForLabel( combinations{comb}, annotStructs{annot}, 90 );
				
			if(~isempty(labelString) )
				classNo = findClassNo( labelString );
				labels(comb) = classNo;
			end
			
		end	
	end
	
end



end

function classNo = findClassNo( className )
	tmp = load('Labels\labels.mat');
	labels = tmp.labels;
	classNo = find(ismember(labels, className));
end

function label = compareForLabel(comb, annot, passPercantage)

% Convert annot struct type to matrix
sketch = annot.sketch;
annotCoords = []; 

for strk = 1 : length(sketch)
	annotCoords = [annotCoords ; sketch(strk).coords];
end

% Compare coordinate matrices
combCoords = comb.coords; 

% Eliminate redundant points (generates bug... I spent 5 hours to find it.)
combCoords = unique(combCoords, 'rows');
annotCoords = unique(annotCoords, 'rows');


%If their size does not match as percentage, do not bother to compare
if( length(combCoords) > length(annotCoords) )
	largestLength = length(combCoords);
else
	largestLength = length(annotCoords);
end


%% Check matching

% Count matching points
matchCount = 0;

for combPntIdx = 1 : size(combCoords,1)
	for annotPntIdx = 1 : size(annotCoords,1)
		
		if( annotCoords(annotPntIdx,1) == combCoords(combPntIdx,1) ...
				&& 	annotCoords(annotPntIdx,2) == combCoords(combPntIdx,2) )
			matchCount = matchCount + 1;
		end
		
	end
end

% Decide
matchPerc = (matchCount / largestLength) * 100;

if(matchPerc > passPercantage)
	label = annot.label;
else
	label = '';
end


end

function annotationStructs = loadAnnotFiles(annotFiles)
fprintf('Loading annotation files...\n');

if(isempty( annotFiles ) )
	annotationStructs = {};
else
	
	for file = 1 : length(annotFiles)
		tmp = load( annotFiles{file} );
		annotationStructs{file} = tmp.annotData;
	end
	
end
end

% Fragments a sketch with dp algo.
% Fragments a sketch with dp algo.
function [ parsed ] = fragmentFile( strokes, DPthreshold )

fprintf('Fragmenting sketch...\n');

%% Fragment file
fragmented = fragmentDP(strokes,DPthreshold);

%% Parse
index = 1;
for i=1:length(fragmented)
	tmp = parseStrokeToPrimitive(fragmented{i}, i);
	
	for j=1:length(tmp)
		parsed{index} = tmp{j};
		index = index + 1;
	end
end

end


function [combined] = extractCombinations(parsed, combinationStart, combinationEnd)

%% Get combinations
fprintf('Getting Combinations...\n');

PrimitiveNums = [];

index = 1;
for combNo = combinationStart-1:combinationEnd-1
	startPnt = 1;
	while(true)
		if(startPnt + combNo <= length(parsed))
			combinations{index,1}  = startPnt:combNo+startPnt;
			
			PrimitiveNums(index) = combNo+1;
			
			index = index + 1;
			startPnt = startPnt + 1;
			
		else
			break;
		end
	end
end



%% Form a structure from combinations
% Strucutre:
% All primitives as one stroke

fprintf('Combining primitives...\n');

% For all combinations
for combNo = 1:length(combinations)
	
	% Combine the coordinates for those primitives
	combinedCoords = [];
	combinedIDs = [];
	for combIndex = combinations{combNo}(1):combinations{combNo}(length(combinations{combNo}))
		combinedCoords = [combinedCoords ;parsed{1,combIndex}.coords];
		combinedIDs = [combinedIDs (parsed{1,combIndex}.strokeID * ones(1,parsed{1,combIndex}.npts))];
	end
	
	% I just need coordinates for IDM features, so I dont use time
	% attribute. The other reason that I dont use time is that when the
	% time attributes are recorded, it records the in-stroke time. Which
	% means time attribute indicates the time passed between the start of
	% stroke, not start of the whole sketch.
	
	combined{combNo} = struct(...
		'coords', [], ...
		'strokeIDs', []);
	
	combined{combNo}.coords     = combinedCoords;
	combined{combNo}.strokeIDs  = combinedIDs;
	combined{combNo}.primNo	    = PrimitiveNums(combNo);
end

end






