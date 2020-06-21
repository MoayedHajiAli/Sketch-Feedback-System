%
% parseStrokeToPrimitive.m
%
% Kemal Tugrul Yesilbek
%
% Takes a stroke which already have the dppoints in the sturcture
% and make each of those primitves a different structure. Which means
% this function makes each primitve a different stroke. In another words,
% this function just parses the structure.
% Another important point is that this function assigns a GUID to each
% stroke and adds to those parsed primitives an attribute that points the
% stroke it belongs to with adding stroke GUID to its structure. GUID
% should be passed to the function. An intutive approach is to pass the
% stroke number to function as GUID
%
% Input:
%
% GUID to define the stroke
%
% Stroke with structure of:
% fragmented{1, 4}.dppoints, 
% fragmented{1, 4}.coords, 
% fragmented{1, 4}.times, 
% fragmented{1, 4}.primids, 
% fragmented{1, 4}.primtypes, 
% fragmented{1, 4}.corners, 
% fragmented{1, 4}.npts, 
% fragmented{1, 4}.nprims
%
% Output:
% Returns a structure:
% coords,
% times
% strokeID
% npts

function [ prims ] = parseStrokeToPrimitive(stroke, StrokeID)
    
    % Do for all primitives
    prims = [];
    for i=1:length(stroke.dppoints)-1
        
        % Define the starting and ending indices of the primitive in the
        % stroke
        startIndex = stroke.dppoints(i);
        endIndex = stroke.dppoints(i+1);
        
        % Create a structure to define primitive
        prims{i} = struct(...
            'coords', [], ...
            'times', [],  ...
            'strokeID', [], ...
            'npts', []);
        
        % Assign the attributes
        prims{i}.coords(:,1)     =  stroke.coords(startIndex:endIndex,1);
        prims{i}.coords(:,2)     =  stroke.coords(startIndex:endIndex,2);
        prims{i}.times           =  stroke.times(startIndex:endIndex);
        prims{i}.strokeID        =  StrokeID;
        prims{i}.npts            =  (endIndex - startIndex)+1;
        
    end

end










