function annotated = annot_v2(trf)

close all;
clc;

%% Parse file path

% Extract selected file name and root folder path
delimeterPlaces = strfind(trf, '\');
global rootFolder; global fileName;
rootFolder = []; fileName = [];
rootFolder = trf(1 : delimeterPlaces(end) );
fileName = trf(delimeterPlaces(end) + 1 : end);

%% Get annotations and see if there are annotations
getAnnotations;
global annotFilePaths;
dpThreshold = 10;

%% Get annot dp threshold
if(length(annotFilePaths) > 0)
	tmp = load( annotFilePaths{1} );
	annotData = tmp.annotData;

	if(isempty(annotData.DPthreshold))
		dpThreshold = 10;
	else
		dpThreshold = annotData.DPthreshold;
	end

	fprintf('DP Threshold is set to %d automatically...\n', dpThreshold);
end

%% Load Current File
global strokes;
strokes = [];
strokes = get_strokes(trf);%fileName);
strokes = douglas_peucker(strokes, dpThreshold);

theGUI;

end


function theGUI


getAnnotations;

%% Get screen resolution
screenResolution = get(0,'screensize');
screenResolution = [screenResolution(3) ; screenResolution(4)];

%% Create and open the window
global hs;
hs = addcomponents(screenResolution);
hs.fig.Visible = 'on';

%% Initialize sketcher
% Sketching is ended when user clicks "select"
global annotSketch;
annotSketch = [];

global isMouseDown;
isMouseDown = false;

global strokes;
global annotFilePaths;
sketcher_v2( strokes, annotFilePaths );

% Show and wait user response
shg;
uiwait;


end

function getAnnotations
%% Load annotations previously made on this file
% Get annotation file names
global rootFolder; global fileName;
global annotFilePaths;
annotFilePaths = [];

annotFilePaths= getFiles(rootFolder, false, ['annot_' fileName]);

for i = 1 : length(annotFilePaths)
	annotFilePaths{i} = [rootFolder '\' annotFilePaths{i}];
end
end

function hs = addcomponents(screenResolution, hs)

%% Options 

% Label list placement
labelList_w = 100;
labelList_h = 450;
labelList_x = screenResolution(1) - labelList_w - 10;
labelList_y = 350;

% Label button placement
labelButton_w = 150;
labelButton_h = 30;
labelButton_x = labelList_x;
labelButton_y = 100;

% Threshold slider placement
thresholdSlider_w = 100;
thresholdSlider_h = 30;
thresholdSlider_x = 30;
thresholdSlider_y = 500;

% Done button placement
doneButton_w = 100;
doneButton_h = 30;
doneButton_x = labelButton_x;
doneButton_y = 10;

% Tool toogle button placement
toolButton_w = 150;
toolButton_h = 60;
toolButton_x = 30;
toolButton_y = 300;

% Selection text placemenet
toolText_w = 150;
toolText_h = 20;
toolText_x = toolButton_x;
toolText_y = toolButton_y - 10 - toolText_h;

% Select button placement
selectButton_w = doneButton_w;
selectButton_h = doneButton_h * 3;
selectButton_x = doneButton_x;
selectButton_y  = 200;

% Clear button placement
clearButton_w = selectButton_w;
clearButton_h = 20;
clearButton_x = selectButton_x;
clearButton_y = selectButton_y -  50;


%% Add components, save handles in a struct

hs.fig = figure('Visible','off',...
	'Resize','off',...
	'units', 'normalized',...
	'outerposition',[0 0 1 1],...
	'Tag','fig');

shg;

hs.labelButton = uicontrol(hs.fig,...
	'Position',[labelButton_x labelButton_y labelButton_w labelButton_h],...
	'String','Label',...
	'Tag','labelButton',...
	'Callback',@labelClicked);

hs.labelList = uicontrol( hs.fig,...
	'Style','listbox',...
	'Max',1,'Min',0,... 
	'Position',[labelList_x labelList_y labelList_w labelList_h]);

% Load class names
tmp = load('labels.mat');
labels = tmp.labels;
set(hs.labelList, 'String', labels);

hs.thresholdSlider = uicontrol(hs.fig,...
	'Style','slider',...
	'Min',0,'Max',150,'Value',10,...
	'SliderStep',[1/150 1/150],...
	'Tag','thresholdSlider',...
	'callback', @sliderCallback,...
	'Position',[thresholdSlider_x thresholdSlider_y thresholdSlider_w thresholdSlider_h]);
	
	hLstn = handle.listener(hs.thresholdSlider,'ActionEvent',@sliderCallback);


hs.doneButton = uicontrol(hs.fig,...
	'Position',[doneButton_x doneButton_y doneButton_w doneButton_h],...
	'String','Done',...
	'Tag','doneButton',...
	'Callback',@doneClicked);

hs.clearButton = uicontrol(hs.fig,...
	'Position',[clearButton_x clearButton_y clearButton_w clearButton_h],...
	'String','Clear',...
	'Tag','clearButton',...
	'Callback',@clearClicked);

hs.toolButton = uicontrol(hs.fig,...
	'Style','togglebutton',...
	'String','Toogle Selection',...
	'Value',0,...
	'Callback', @toolClicked,...
	'Position', [toolButton_x toolButton_y toolButton_w toolButton_h]);

hs.toolText = uicontrol(hs.fig,...
	'Style','text',...
	'String','Free-Sketching',...
	'Position',[toolText_x toolText_y toolText_w toolText_h] );

hs.selectButton = uicontrol(hs.fig,...
	'Position', [selectButton_x selectButton_y selectButton_w selectButton_h],...
	'String','Select',...
	'Tag','selectButton',...
	'Callback',@selectClicked);

end

%% UI Callbacks
function toolClicked(~, ~)

global hs;	
val = get(hs.toolButton, 'Value');

if(val == 0)
	set(hs.toolText, 'String', 'Free-Sketching');
else
	set(hs.toolText, 'String', 'Pinpointing');
end

end

function sliderCallback(~,~)

global hs;	
val = get(hs.thresholdSlider, 'Value');
val = round(val);


global strokes;
strokes = douglas_peucker(strokes, val);

% Plot annotations
cla;
global annotFilePaths;
getAnnotations;

sketcher_v2( strokes, annotFilePaths);

global annotSketch;
annotSketch = [];


global threshold;
threshold = val;

end

function selectClicked(hObject, callbackdata)
	
	global annotSketch;
	annotSketch = abs(annotSketch);		% The figure y is neg. so make it pos.

	%% Find Corners Within Annotation
	
	global hs;
	global strokes;
	annotSketch
	if( get(hs.toolButton, 'Value') == 0)
		cornersInside = findCornersWithinFreeDrawing( annotSketch, strokes );
	else
		cornersInside = findCornersWithinPinpoint( annotSketch, strokes );
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
	global sw;
	sw = extractSketchWord( strokes );
	
	%% Show annotated sketch word
	lightenSelectedSketch(sw);
	
end

function doneClicked(hObject, callbackdata)
uiresume; % Close UI
close all;
clc;
end

function clearClicked(hObject, callbackdata)

% Plot annotations
cla;
global annotFilePaths;
global strokes;
getAnnotations;

sketcher_v2( strokes, annotFilePaths);

global annotSketch;
annotSketch = [];

end

function labelClicked(hObject, callbackdata)

global hs;	
labelIdx = get(hs.labelList, 'Value');
tmp = get(hs.labelList,'String');
label = tmp{labelIdx};

%% Save the annotation
% Bundle the annotation
global sw;
annotData.sketch = sw;
annotData.label = label;
global threshold;
annotData.DPthreshold = threshold;
annotData.date = datestr(now,'mmmm dd, yyyy HH:MM:SS.FFF AM');

% Save annotation to file
global rootFolder; global fileName;
filename = [rootFolder '\' 'annot_' fileName '_' label '_' datestr(now,'mmmm_dd_yyyy_HH_MM_SS_AM') '.mat'];
save(filename, 'annotData');

% Plot annotations
cla;
global annotFilePaths;
global strokes;
getAnnotations;

sketcher_v2( strokes, annotFilePaths);

global annotSketch;
annotSketch = [];

end

%% Inner working

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

% Plots a mark on the given corners
function lightenSelectedCorners(corners, sketch)

hold on;
for cor = 1 : length(corners)
	dpIdx = sketch( corners(cor,1) ).dppoints( corners(cor,2) );
	cornerCoord = sketch( corners(cor,1) ).coords( dpIdx, :);
	plot( cornerCoord(1), - cornerCoord(2), 'cx' );
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

function lightenSelectedSketch(sk)

for k = 1:length(sk)
	c = sk(k).coords;
	for pnt = 1 : length(c)
		plot( c(pnt,1), -c(pnt,2), 'ks');
	end
end

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
