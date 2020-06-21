% combinationTest.m
%
% Primitive combination test
% Combines the primitives
%
% Kemal Tugrul Yesilbek

%% Initialization
close all;
clear all;
clc;
workspace;

%% Options
DPTreshold = 11;

% This param. controls the #primitive lower-limit. If this is 2, then the
% combinations contains only 1 primitive wont be in the result. If this is
% n, then the combinations contanins 1:n-1 primitives wont be in the
% result.
combinationStart = 2;

% This param. controls the #primitive upper-limit. If this is 5 then the
% combinations contains more then 5 wont be in the result. If this is n,
% then the combinations contains n+1:Inf primitives wont be in the result.
combinationEnd = 11;

isPlot = true; % Is plot final combinations
isSave = true; % save the file?


%%  Select .xml
[fileName,pathName] = uigetfile('*.xml','Select the file to be fragmented');
file = [pathName fileName];

%% Read file
fprintf('Reading Sketch...\n');
strokes = read_sketch(file);

%% Fragment file
fragmented = fragmentDP(strokes,DPTreshold);

figure; hold on;
colors = lines( length(fragmented) );
for i = 1 : length(fragmented)
	for pnt = 1: length(fragmented{i}.coords)
		plot( fragmented{i}.coords(pnt,1),  -fragmented{i}.coords(pnt,2), '.-', 'color', colors(i,:) );
	end

	for dp = 1: length(fragmented{i}.dppoints)
		dpcoord =  fragmented{i}.coords(   fragmented{i}.dppoints(dp) , : );

		plot( dpcoord(1), -dpcoord(2) , 'ks' );
	end

end


%% Parse
index = 1;
for i=1:length(fragmented)
   tmp = parseStrokeToPrimitive(fragmented{i}, i);

   for j=1:length(tmp)
       parsed{index} = tmp{j};
       index = index + 1;
   end
end

%% Get combinations
fprintf('Getting Combinations...\n');

index = 1;
for combNo = combinationStart-1:combinationEnd-1
    startPnt = 1;
    while(true)
        if(startPnt + combNo <= length(parsed))
            combinations {index,1}  = startPnt:combNo+startPnt;
            index = index + 1;
            startPnt = startPnt + 1;
        else
            break;
        end
    end
end

% Visualize the combinations with a histogram
combHist = zeros(1,length(parsed));
for i=1:length(combinations)
   combHist(length(combinations{i})) = combHist(length(combinations{i})) + 1;
end

figure;
bar(combHist); xlabel('#Primitives'); ylabel('#Occurance');


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
end

%% Visualize the combined ones
if(isPlot)
	fprintf('Printing sketchs...\n');



	parfor i = 1:length(combined)
		fprintf('%d of %d...\n', i, length(combined));
		f = figure('visible','off');
	   % Change the color of primitive
		r = (1-0).*rand(3,1) + 0;

		% Print points
		for pntNo = 1:length(combined{i}.coords) % For each point

			plot(combined{i}.coords(pntNo,1),-combined{i}.coords(pntNo,2),...
				'color',r,'marker','.');

			if(pntNo ~= 1)
				line(  [combined{i}.coords(pntNo,1), combined{i}.coords(pntNo-1,1)], ...
						[-combined{i}.coords(pntNo,2), -combined{i}.coords(pntNo-1,2)], 'color', r );
			end

			hold on;
		end

		title(['#Primitives: ' num2str(length( unique(combined{i}.strokeIDs)))]);

		img = strcat(['CombImg\comb_' num2str(i) '.png']);
		saveas(f,img);

		close;
	end
end


if(isSave)
	save([fileName(1:end-4) '_comb.mat'],'combined');
end
