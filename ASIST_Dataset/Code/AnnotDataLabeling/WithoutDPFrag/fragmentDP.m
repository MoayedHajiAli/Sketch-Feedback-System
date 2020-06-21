% fragmentDP.m
%
% Kemal Tugrul Yesilbek
%
% Fragments the sketch with Douglas-Peucker algorithm
%
% Input:
% <stroke> <stroke> ... <stroke>
%
% <stroke> struct:
% coords
% times
% primids...
%
% (Same format as read_sketch returns)

function [ fragmented ] = fragmentDP( strokes, DPTreshold )


% Fragment each stroke seperately
for i=1:length(strokes)
    current = strokes(i);
    
    fragmented{i} =   struct(...
        'coords', [], ...       % coordinates
        'times', [], ...        % times of each point in array
        'primids', [], ...      % pirimitive id of each point
        'primtypes', [] , ...   % types of the primitives
        'corners', [], ...      % index of connections of two primitives
        'npts', [], ...         % number of points
        'nprims', []);          % distinct primitive ids
    
    fragmented{i} = douglas_peucker(current, DPTreshold);
end


end

