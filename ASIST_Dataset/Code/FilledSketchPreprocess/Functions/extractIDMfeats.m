function [ IDMvector ] = extractIDMfeats( sketchData )
% sketchData:
% -- coords
% -- strokeIDs

% IDM Parameters
IDMParams = [50 10 4];

%% Parse sketch data into stroke structure
uniqueStrokeIDs = unique(sketchData.strokeIDs);

for strkID = 1 : length(uniqueStrokeIDs)
    
    idx = find( sketchData.strokeIDs == uniqueStrokeIDs(strkID) );
    coords = sketchData.coords( idx, : );
    
    strokes(strkID) = struct('coords', []);
    strokes(strkID).coords = coords;
end

%% Extract feature vector
IDMvector = idm(strokes, IDMParams(1), IDMParams(2), IDMParams(3));

end

