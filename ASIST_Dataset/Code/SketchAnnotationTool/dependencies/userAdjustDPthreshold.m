% User adjust the threshold with keyboard. In value change, the result is
% displayed to the user
function [ dpthreshold ] = userAdjustDPthreshold( fileName )

%% Options
close all;
clc;

dpthreshold = 3;

%% Load Current File
strokes = get_strokes(fileName);

%% Fragment, show, wait for input
while(true)
	% Fragment
	if(dpthreshold < 0)
		dpthreshold = 0;
	end
	
	strokes = douglas_peucker(strokes, dpthreshold);
	
	% Plot
	figure;
	for k=1:length(strokes)
		plot_stroke(strokes(k));
	end
	title(['DPThreshold: ' num2str(dpthreshold)]);
	
	% Wait for input
	DPadjust;
	global adjustDecision;
	
	if( adjustDecision > 0) % Increase 
		close;
		dpthreshold = dpthreshold + adjustDecision;
	elseif( adjustDecision < 0) % Decrease
		close;
		dpthreshold = dpthreshold + adjustDecision;
	elseif( adjustDecision == 0) % Done
		close;
		break;
	end
end



end

