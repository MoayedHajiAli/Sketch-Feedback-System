function  annotChecker_main( trf )

%% Parse file path

% Extract selected file name and root folder path
delimeterPlaces = strfind(trf, '\');
rootFolder = []; fileName = [];
rootFolder = trf(1 : delimeterPlaces(end) );
fileName = trf(delimeterPlaces(end) + 1 : end);

%% Load Current File
strokes = [];
strokes = get_strokes(trf);
strokes = douglas_peucker(strokes, 10);

%% Get annotation file names
annotFilePaths= getFiles(rootFolder, false, ['annot_' fileName]);

for i = 1 : length(annotFilePaths)
	annotFilePaths{i} = [rootFolder '\' annotFilePaths{i}];
end

%% Visualize Sketch
fig = figure('visible','off');
hold on;
% Plot Current Sketch
for k=1:length(strokes)
	plot_stroke(strokes(k));
end

% Plot annotations
for i = 1 : length(annotFilePaths)
	plotAnnotFile( annotFilePaths{i} );
end

% Add title
title(fileName);

% Save to the disk
img = [rootFolder '\vis_'  fileName '.png'];
saveas(fig,img);


end


% Loads and plots the coordinates in annotation file
function plotAnnotFile( annotFile )
	tmp = load(annotFile);
	annotData = tmp.annotData;
	annotSketch = annotData.sketch;
	annotLabel = annotData.label;
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
	
	% Place label
	[labelPlaceX, labelPlaceY] = getSketchMean(annotSketch);
	text( labelPlaceX, -labelPlaceY, annotLabel );
	
end

function [meanX, meanY] =  getSketchMean(annotSketch)

X = []; Y =[];

for strk = 1 : length( annotSketch )
	coord = annotSketch(strk).coords;
	
	for pnt = 1 : length(coord)
		X = [X coord(pnt,1)];
		Y = [Y coord(pnt,2)];
	end
end

meanX = mean(X);
meanY = mean(Y);


end









