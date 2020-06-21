% processFilled.m
%
% Kemal Tugrul Yesilbek
% June 2015
%
% Processes the filled sketch, and returns clearer coordinates. Here is how
% it works:
%
%
% 1. Find convex hull
%
% 2. Construct a graph for sketch
%
% 2.a. Find 10 N.N. for each point
%
% 2.b. Connect each point with their 10 N.N.s and associate weights with
% distances
%
% 3. Find shortest path for each consequtive convex hull points (i.e. 1-2,
% 2-3, 3-4, ..., N-1)
%
% 4. Accept the nodes in path as the new coordinates
%
%

function [ newCoords ] = processFilled( coords )

%% Find the convex hull points of the sketch (1st step)
convIdx = convhull( coords(:,1), coords(:,2) ); % Conv hull returns ordered points

%% Construct a graph (2nd Step)
graph = constructGraph( coords, convIdx );

if(size(graph,1) ~= size(graph,2))
    fprintf('There is a problem with graph...\n');
    newCoords = coords;
    return;
end

%% Find shortest paths (3rd Step)
pathIdx = [];
for i = 2 : length(convIdx)
    [dist,path,pred] = graphshortestpath(graph, convIdx(i-1), convIdx(i));
    pathIdx = [pathIdx, path];
end
[dist,path,pred] = graphshortestpath(graph, convIdx(end), convIdx(1));
pathIdx = [pathIdx, path];


%% Accept the nodes in path as the new coordinates (4th Step)
newCoords = coords( pathIdx, :);


%% Visualize results
% printCoords(coords);
% hold on;
% for i = 1:length(convIdx)
%     idx = convIdx(i);
%     plot( coords(idx,1), coords(idx,2), 'ro');
% end
% 
% for i = 1:length(pathIdx)
%     idx = pathIdx(i);
%     plot( coords(idx,1), coords(idx,2), 'k*');
% end
% 
% drawnow;
% 
% waitforbuttonpress;


end


function graph = constructGraph( coords, convIdx )

    N = 10;
    V1 = []; V2 = []; E = [];
    
    %% Construct Local Connections
    for pnt = 1 : length(coords)
        % Find 10 N.N.s
        [nns, nnsDist] = findNNs(coords, pnt, N);
        
        % Construct local pairs
        V1 = [V1; ones( length(nns), 1 ) * pnt];
        V2 = [V2; nns];
        E = [E; nnsDist];
    end
    
    %% Construct Graph
    graph = sparse( V1, V2, E );
    

end


function [nns, nnsDist] = findNNs(coords, pivotIdx, N)

    % Check input
    if(length(coords) <= N)
        N = length(coords);
    end

    % Calculate pairwise distances of points
    dist = ones( length(coords), 1) * -1;
    for pnt = 1 : length(coords)
        if( pnt == pivotIdx)
            dist(pnt) = +Inf;
        else
            dist(pnt) = pdist( [coords( pivotIdx, :) ; coords( pnt, :)] );
        end
    end
    
    % Pick top N instances
    nns = []; nnsDist = [];
    for n = 1 : N
        [C, I] = min( dist );
        nns = [nns ; I];
        nnsDist = [nnsDist ; C];
        dist(I) = +Inf;
    end

end









































