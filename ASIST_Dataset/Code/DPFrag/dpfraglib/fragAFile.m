% fragAFile.m
%
% Uses DPFrag with the pre-automized sketches to fragment and form the
% sequencial partitions the given sketch.
%
% Input:
% filePath: Path of the file to be fragmented
% debugLevel: Level of debug;
%   0: No debug
%   1: Just fragmentation view
%   2: Full debug
%
% Output:
% xml: XMLs of the partitiones
function xml = fragAFile( filePath, debugLevel)

    strokes = read_sketch(filePath);
    stroke_num = length(strokes);
    indices = get_corners(filePath);
    
    if(debugLevel > 0)
        hold off;
        figure;
        title(filePath);
        hold on;
    end
    
    
    
    p = 1;
    c = 1;
    for j=1:stroke_num
        cc=hsv(stroke_num*length(indices{1,j})); %assign different colors to parts btx two detected points
        
        for k=1:length(indices{1,j})-1
            %plot parts btw detected points each with different lines
            %whole constitute the actual drawing
            if(debugLevel == 1 || debugLevel == 2)
                %plot(strokes(1,j).coords(indices{1,j}(1,k):indices{1,j}(1,k+1),1), strokes(1,j).coords(indices{1,j}(1,k):indices{1,j}(1,k+1),2), 'color', cc(c,:), 'Marker','*');
            end
            X{c} = strokes(1,j).coords(indices{1,j}(1,k):indices{1,j}(1,k+1),1);
            Y{c} = strokes(1,j).coords(indices{1,j}(1,k):indices{1,j}(1,k+1),2);
            T{c} = strokes(1,j).times(indices{1,j}(1,k):indices{1,j}(1,k+1),1);
            c = c + 1;
        end
        
        if(debugLevel > 0)
            axis equal;
        end
        p = p + 1;
        
        if(debugLevel == 1 || debugLevel == 2)
            %plot detected points with black circles
            plot(strokes(1,j).coords(indices{1,j},1), strokes(1,j).coords(indices{1,j},2), 'blacko','LineWidth',3);
        end
    end
    
    % this is the part where combinations of 2,3,4.. subparts are found
    % first param: primitive number means subparts with 2 primitive will be shown 
    % second param: different parts (primitives) in sketch
    % X,Y : the coordinates of these primitives

    % get k combinations sorted in time for the xml specified above
    count = 1;
    
    combinationLimit = 7;
    
    % c = #primitives
    for k=1:c-1
        
        [imcell] = get_combinations(k,c,X,Y,T,debugLevel==2);
        
        for i=1:length(imcell)
            xml{count} = imcell{i};
            count = count + 1;
        end
        
        if( k == combinationLimit)
            break;
        end
    end

end

