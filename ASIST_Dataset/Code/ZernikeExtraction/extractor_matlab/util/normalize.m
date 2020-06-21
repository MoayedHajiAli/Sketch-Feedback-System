function [coords coords_per_stroke] = normalize(coords_per_stroke)
    
    coords = perstroke2coords(coords_per_stroke);     
    centroid = mean(coords);    
    stdev = std(coords,0,1);
    if(stdev(1,1)==0)
        stdev(1,1)=1;
    end
    if(stdev(1,2)==0)
        stdev(1,2)=1;
    end
    for i=1:length(coords_per_stroke)
        scoords = coords_per_stroke{i};
        scoords = scoords - repmat(centroid, size(scoords,1),1);
        scoords = scoords .* repmat(1./stdev, size(scoords,1),1);
        coords_per_stroke{i} = scoords;
    end
    coords = perstroke2coords(coords_per_stroke); 

end

function coords = perstroke2coords(coords_per_stroke)
    coords = []; 
    for i=1:length(coords_per_stroke)
        coords = [coords;coords_per_stroke{i}];
    end  
end
