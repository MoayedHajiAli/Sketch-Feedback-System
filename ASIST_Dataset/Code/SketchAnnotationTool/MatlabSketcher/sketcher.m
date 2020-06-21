% sketcher.m
%
% Kemal Tugrul Yesilbek
%
% Body Gesture Retrieval via Sketch Representation
% March 2014

function [ sketch ] = sketcher( strokes, annotFileNames)

%
% Warning !!!!!!
% This code will only run on Matlab versions prior to R2014b. This is because
% the graphic handles changed with R2014b and is not compatible with this code.
%
% Collects the sketch point data for specified number of frames
%
% How to use:
%
% Starts to collect point as mouse (tip touches) pressed
% Stops to collect point as mouse (tip hovers) unpress
% Each stroke and frame is collected seperately
% Hit a keyboard key to finish sketching for that frame
%
% IN:
% strokes : sketch as format returned by xml_load.m

%nodePoints(frameNo, clickNo, 1) = mousePosX;
%nodePoints(frameNo, clickNo, 2) = mousePosY;
%[ frameNo , X , Y ]
nodePoints = [0 0 0];

% Mouse positions will come from global variables workspace
global mousePosX;
global mousePosY;
global isSketching;
global isSketchDone;
global isMouseUp;

% Record the points for all frame
isSketching = false;

% Plot Current Sketch
for k=1:length(strokes)
	plot_stroke(strokes(k));
end

% Plot annotations
for i = 1 : length(annotFileNames)
	plotAnnotFile( annotFileNames{i} );
end

% Open up a new figure for new frame
hold on;

%set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.

set (gcf, 'windowbuttonmotionfcn', @mouseMove)
set (gcf, 'windowbuttondownfcn'  , @mouseDown)
set (gcf, 'windowbuttonupfcn'      , @mouseUp)
set (gcf, 'windowkeypressfcn'     , @keyPress)


% Record skech
isSketchDone = false;
strokeNo = 1;
pointID = 1;
while(~isSketchDone)
	
	if(isSketching)
		plot(mousePosX,mousePosY,'r.');
		sketch{strokeNo}(pointID,:) = [mousePosX mousePosY];
		pointID = pointID + 1;
	end
	
	if(isMouseUp) % The tip-up frame
		isMouseUp = false;
		strokeNo = strokeNo + 1;
		pointID = 1;
	end
	
	% Show figure title and refresh
	%title(['Draw the frame: ' num2str(frameNo) '/' num2str(numFrameToDraw)]);
	%shg;
end

end

% Loads and plots the coordinates in annotation file
function plotAnnotFile( annotFile )
	tmp = load(annotFile);
	annotData = tmp.annotData;
	annotSketch = annotData.sketch;
	
	
	hold on;
	
	for strk = 1 : length( annotSketch )
		coord = annotSketch(strk).coords;
		
		for pnt = 1 : length(coord)
			
			if(pnt == 1)
				plot( coord(pnt,1), -coord(pnt,2), 'gs');
			else
				plot( coord(pnt,1), -coord(pnt,2), 'gs');
				line([coord(pnt-1,1), coord(pnt,1)], [-coord(pnt-1,2), -coord(pnt,2)], 'Color', 'g');
			end
		end
	end
	
	
end
























