% annot.m
%
% Kemal Tugrul yesilbek
% March 2015
%
% In:
% trf: cell of .xml paths
%
% Out:
%
% Use:


function annotated = annot(trf, selectedThreshold)

close all;
clc;

annotated = annot_main(trf, selectedThreshold);

end
 
% trf = cell of .xml paths
% i = from which file to start (KLUDGE)
function strokes = annot_main(trf, selectedThreshold)

	%% Parse file path
	
	% Extract selected file name and root folder path
	delimeterPlaces = strfind(trf, '\');
	rootFolder = trf(1 : delimeterPlaces(end) );
	fileName = trf(delimeterPlaces(end) + 1 : end);
	
	%% Load Current File 
	strokes = get_strokes(trf);%fileName);
	strokes = douglas_peucker(strokes, selectedThreshold);
	
	%% Load annotations previously made on this file
	% Get annotation file names
	annotFilePaths = getFiles(rootFolder, false, ['annot_' fileName]);	 
	for i = 1 : length(annotFilePaths)
		annotFilePaths{i} = [rootFolder '\' annotFilePaths{i}];
	end
	
	%% Open Sketcher
	[ annotation ] = sketcher(strokes, annotFilePaths);	% Returns the annotation sketch
	
	annotation = mergeStrokes( annotation );
	annotation = abs(annotation);		% The figure y is neg. so make it pos.
	
	
	
	%% Ask user the annotation type
	getAnnotType; % The output will be in the base workspace as var. with name "annotType"
						  % (AnnotType) 1 = Free Frawing || 2 = Pinpointing
	
	%% Find Corners Within Annotation
	global annotType;
	if(annotType == 1)
		cornersInside = findCornersWithinFreeDrawing( annotation, strokes );
	else
		cornersInside = findCornersWithinPinpoint( annotation, strokes );
	end
	
	% No corners found exception
	if(isempty( cornersInside ))
		warning('No Corners Found!!!');
		errorOccured;
		annot_main(trf, selectedThreshold);
	end
	
	%% Post-process the Selection
	% Remove the selected corner if its the only selected corner in its
	% stroke
	cornersInside = removeSingleCorner(cornersInside);
	
	% Lighten up the selected corners!
	lightenSelectedCorners(cornersInside, strokes);
	
	%% Extract
	% Label points that belong to annotated part
	strokes = markSketchWord(cornersInside, strokes);
	
	% Extract annotated part
	sw = extractSketchWord( strokes );
	
	%% Show annotated sketch word
	% Plot extracted sketch word
	figure; 
	for k=1:length(sw)
		plot_stroke(sw(k));
	end
	title('Extracted Part');
	
	%% Ask user for confirmation
	sketchPartConfirm; % The output will be in the base workspace as var. with name "isConfirmed"
	
	% If user did not confirmed the selection, rerun current file for
	% annot.
	global isConfirmed;
	if(~isConfirmed)
		close all;
		clc;
		annot_main(trf, selectedThreshold);
		return;	% Matlab has a wierd recursion mechanism o.O
	end
	
	%% Ask for label
	labelSelection; % The output will be in the base workspace as var. with name "selectedLabel"
	
	%% Save the annotation
	% Bundle the annotation
	annotData.sketch = sw;
	global selectedLabel; annotData.label = selectedLabel;
	annotData.DPthreshold = selectedThreshold;
	annotData.date = datestr(now,'mmmm dd, yyyy HH:MM:SS.FFF AM');
	
	% Save annotation to file
	filename = [rootFolder '\' 'annot_' fileName '_' selectedLabel '_' datestr(now,'mmmm_dd_yyyy_HH_MM_SS_AM') '.mat'];
	save(filename, 'annotData');
	
	%% Ask for new annotation for this file
	annotateMore;
	
	global isAnnotateMore;
	if(isAnnotateMore)
		close all;
		clc;
		annot_main(trf, selectedThreshold);
	end
	
	close all;
end


 
% Gives only the sketch word
function sketch = extractSketchWord( sketch )
	
	% Remove the points marked as non-sketch word
	for strk = 1 : length(sketch)
		
		pnt = 1;
		while(true)
			 
			if(pnt > length(sketch(strk).times))
				break;
			end
			
			if( sketch(strk).primtypes(pnt) == 0 )
				
				sketch(strk).coords(pnt,:) = [];
				sketch(strk).times(pnt,:) = [];
				sketch(strk).primtypes(pnt,:) = [];
				
				sketch(strk).npts = sketch(strk).npts - 1;
				
				pnt = 0;
			end
			
			pnt = pnt + 1;
		end
	end
	
	% Remove strokes that does not have any points left in it
	strk = 1;
	while(true)
		
		if( strk > length(sketch) )
			break;
		end
		
		if( sketch(strk).npts == 0 )
			sketch(strk) = [];
			strk = 0;
		elseif( sketch(strk).npts < 0 )
			assert('SMT is wrong! #1');
		end
		
		strk = strk + 1;
	end
	
	% Remove dppoints
	for strk = 1 : length(sketch)
		sketch(strk).dppoints = [];
	end
	
end

% Marks the sketch points as s-w or non-s-w
function sketch = markSketchWord( corners, sketch )

% Initialize sketch's primtypes
for strk = 1 : length(sketch)
	sketch(strk).primtypes = zeros( sketch(strk).npts,1 );
end

corners = sortrows( corners, 1 ); % Sort sketch with stroke no

% Label sketch points
for cor = 1 : length(corners)
	
	strkNo = corners(cor, 1);
	
	if(cor == length(corners))	
		startIdx = sketch(strkNo).dppoints( corners(cor - 1, 2) );
		endIdx = sketch(strkNo).dppoints( corners(cor, 2) );
		
		sketch(strkNo).primtypes( startIdx : endIdx ) = 1;
		continue; 
	end
	
	if( corners(cor, 1) == corners(cor + 1, 1) )
		
		startIdx = sketch(strkNo).dppoints( corners(cor, 2) );
		endIdx = sketch(strkNo).dppoints( corners(cor+1, 2) );
		 
		sketch(strkNo).primtypes(startIdx : endIdx) = 1;	
		
	else
		startIdx = sketch(strkNo).dppoints( corners(cor-1, 2) ); 
		endIdx = sketch(strkNo).dppoints( corners(cor, 2) );
		 
		sketch(strkNo).primtypes(startIdx : endIdx) = 1;	
	end
	
	
end

end

% Removes the selection of strokes with single corner
function corners = removeSingleCorner(corners)

% Find max. stroke
[maxStroke, ~] = max( corners(:,1) );

% Count stroke corners
count = zeros( maxStroke, 1 );
for i = 1 : length(corners)
	count( corners(i,1) ) = count( corners(i,1) ) + 1;
end

% Remove corners that has 0 count
cor = 1;
for strk = 1 : maxStroke
	
	while(true)
		if(cor > length(corners) )
			cor = 1;
			break;
		end
		
		% Exception Handling
		if(size(corners,1) < cor)
			warning('No corners found in a stroke!');
			errorOccured;
		end
		
		if( count(strk) == 1 && corners(cor, 1) == strk)
			corners(cor,:) = [];
			cor = 0;
		end
		
		cor = cor + 1;
		
	end
end
		
		
	


end

% Plots a mark on the given corners
function lightenSelectedCorners(corners, sketch)

hold on;
for cor = 1 : length(corners)
	dpIdx = sketch( corners(cor,1) ).dppoints( corners(cor,2) );
	cornerCoord = sketch( corners(cor,1) ).coords( dpIdx, :);
	plot( cornerCoord(1), - cornerCoord(2), 'cx' );
end

end

% Finds corners within a sketch
% Corner Inside (:,1 ) : stroke No
% Corner Inside (:,2 ) : DPpnt No
function cornersInside = findCornersWithinFreeDrawing( annot, strokes )

cornerIndex = 1;

% Find corner edges ( bounding box )
right = max( annot(:, 1) );
left = min( annot(:, 1) );
up = max( annot(:, 2) );
down = min( annot(:, 2) );

% Find corners within edges
for strk = 1 : length( strokes )  % For each stroke
	
	for cor = 1 : length( strokes(strk).dppoints )
		
		cornerIdx = strokes(strk).dppoints(cor);
		cornerCoord = strokes(strk).coords( cornerIdx, : );
		
		% Within?
		if( cornerCoord(1) > left && cornerCoord(1) < right && ...
				cornerCoord(2) > down && cornerCoord(2) < up)
			
			cornersInside(cornerIndex, 1) = strk;
			cornersInside(cornerIndex, 2) = cor;
			cornerIndex = cornerIndex + 1;
			
		end
		
	end
	
end

if(cornerIndex == 1) % No corners found exception
	cornersInside = {};
end

end

% Finds corners within a sketch
% Corner Inside (:,1 ) : stroke No
% Corner Inside (:,2 ) : DPpnt No
function cornersInside = findCornersWithinPinpoint( annot, strokes )
 

% Find corners closest to pinpoints
for i = 1 : length(annot)
	annot(i,:);
	% Find closest corner
	[minStrkIdx, minDpIdx] = findClosestCorner( annot(i,:), strokes );
	cornersInside(i,1) = minStrkIdx;
	cornersInside(i,2) = minDpIdx; 
	
end

% Remove repetitive entries
cornersInside = unique(cornersInside, 'rows');

end

function [minStrkIdx, minDpIdx] = findClosestCorner( x, sketch )
	
	minDist = +Inf;		minStrkIdx = -1;	minDpIdx = -1;

	for strk = 1 : length(sketch)
		for dp = 1 : length(sketch(strk).dppoints)
			
			dist = pdist( [sketch(strk).coords( sketch(strk).dppoints(dp), :); x] ) ;
			if( dist < minDist)
				minDist = dist;
				minStrkIdx = strk;
				minDpIdx = dp;
			end
			
		end
	end

end

% Merges strokes
% In: coords of points as cell array
function merged = mergeStrokes(sketch)
merged = [];
for strk = 1:length(sketch)
	merged = [ merged ; sketch{strk} ];
end
end
































