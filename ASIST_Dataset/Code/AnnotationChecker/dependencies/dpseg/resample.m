function strokes = resample(strokes, interval, anglemeasure)    
    % ratio: typically 50
    % nn : typically 1,2     
    strokes().resampled = [];       
    for i=1:length(strokes)       
        new_coords = [];
        prev = strokes(i).coords(1,:);
        new_coords = [prev 1 nan nan];        
        for j=2:size(strokes(i).coords,1)
            
            while (get_distance(strokes(i).coords(j,:), prev) > interval)
                angle = atan2(strokes(i).coords(j,2) - prev(2), ...
                              strokes(i).coords(j,1) - prev(1));
                prev = [prev(1) + cos(angle)*interval, ...
                        prev(2) + sin(angle)*interval];
                new_coords = [new_coords; prev j nan nan];
            end
            
        end        
        strokes(i).resampled = new_coords;
    end
    
    strokes = add_npts(strokes);
    strokes = add_angles(strokes, anglemeasure);        
    
end    

function dist = get_outlier(pts)
    centroid = mean(pts.coords,1);    
    diff = pts.coords - repmat(centroid, length(pts.coords), 1);
    dist= max(sqrt(sum(diff.*diff,2)));    
end

function strokes = add_npts(strokes)
    for i=1:length(strokes)
        for j=2:size(strokes(i).resampled,1)-1
            strokes(i).resampled(j,4) = strokes(i).resampled(j,3) - strokes(i).resampled(j-1,3) + ...
                                        strokes(i).resampled(j+1,3) - strokes(i).resampled(j,3);
        end
        % normalize
         strokes(i).resampled(:,4) = strokes(i).resampled(:,4) / mean(strokes(i).resampled(2:end-1,4));
    end
    
    
end

function strokes = add_angles(strokes, anglemeasure)
    nn = anglemeasure;
    for i=1:length(strokes)                
        for j= 1+nn : size(strokes(i).resampled,1)-nn
            pl = strokes(i).resampled(j-nn,[1 2]);
            p0 = strokes(i).resampled(j,[1 2]);
            pr = strokes(i).resampled(j+nn,[1 2]);
            strokes(i).resampled(j,5) = sin(get_angle(pl, p0, pr));
        end
        % normalize
        strokes(i).resampled(:,5) = strokes(i).resampled(:,5) / mean(strokes(i).resampled(1+nn:end-nn,5));
    end    
end

function angle = get_angle(pl, p0, pr)
    angle = lines_exp_angle_nd(2, p0, pl, p0, pr);
end
