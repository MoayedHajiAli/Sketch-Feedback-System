% isFilled.m
%
% Kemal Tugrul Yesilbek
% June 2015
%
% Returns 1 if the sketch with passed coordinates is filled else returns 0
%
% Uses a simple rule for detection: the ratio of number of points and the
% area. If this ratio is above a threshold, it gives true output for that
% sketch.
%

function [ out, ratio ] = isFilled( coords, ratioThreshold, pointThreshold )

    [out, ratio] = v1( coords, ratioThreshold, pointThreshold );
    %[out, ratio] = v2( coords, ratioThreshold, pointThreshold );

end


% Not a good heuristic as the mean dist is not normalized
function [out, meanDist] = v2( coords, distThreshold, pointThreshold )

n = length(coords); % Syntactic sugar

%% Check point number
if( n < pointThreshold )
    % Do not bother, it is not filled
    out = false;
    meanDist = 99999;
    return;
end

%% Calculate the mean-interpoint distance
sumDist = 0;
for p1 = 1 : n
    for p2 = p1+1 : n
        sumDist = sumDist + pdist([coords(p1,:) ; coords(p2,:)]);
    end
end

meanDist = sumDist / ( (n * ( n + 1 ) / 2 ) - n ); % Num dist calc: (n * ( n + 1 ) / 2 ) - n

%% Decide
if( meanDist < distThreshold )
    out = true;
else
    out = false;
end

end


function [out, ratio] = v1( coords, ratioThreshold, pointThreshold )

%% Check point number
if( length(coords) < pointThreshold )
    % Do not bother, it is not filled
    out = false;
    ratio = 0;
    return;
end

%% Calculate the area of the sketch
minX = min(coords(:,1));
maxX = max(coords(:,1));
minY = min(coords(:,2));
maxY = max(coords(:,2));

area = (maxX - minX) * (maxY - minY);

%% Calculate the ratio
numPoints = length(coords);
ratio = numPoints / area;

%% Decide
if(ratio > ratioThreshold)
    out = true;
else
    out = false;
end

end