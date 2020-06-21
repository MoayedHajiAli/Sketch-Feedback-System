function printProblematicSketch( xmlFile, fileName, combinations, labels, destinationFolder )

%% Load sketch file
strokes = read_sketch(xmlFile);

%% Visualize sketch
fig = figure('visible','off');
hold on;
for strk = 1 : length(strokes)
	plot_stroke(strokes(strk));
end
title(xmlFile);


%% Visualize labels

% Parse labels
labelsPosIdx = find(labels ~= -1);

% Visualize pos combinations
hold on;
for lab = 1 : length(labelsPosIdx)
	combIdx = labelsPosIdx(lab);
	plotPiece( combinations{combIdx}, 'cs' );
end

% Type labels
for lab = 1 : length(labelsPosIdx)
	combIdx = labelsPosIdx(lab);
	[labelPosX, labelPosY] = getSketchMean( combinations{combIdx} );
	
	annotLabel = findClassName( labels( labelsPosIdx(lab) ) );
	text( labelPosX, -labelPosY, annotLabel );
end

% Save to the disk
img = [destinationFolder '\' 'DPFragResult_' fileName '.png'];
saveas(fig,img);

close all;

end

function className = findClassName( classNo )
	tmp = load('Labels\labels.mat');
	labels = tmp.labels;
	className = labels{classNo};
end

function plotPiece(sketch, pntStyle)
	coords = sketch.coords;
	for pnt = 1 : length(coords)
		plot(coords(pnt,1), -coords(pnt,2), pntStyle);
	end
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

