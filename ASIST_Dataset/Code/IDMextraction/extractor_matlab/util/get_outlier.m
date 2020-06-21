function dist = get_outlier(coords)

    centroid = get_centroid(coords);    
    diff = coords - repmat(centroid, length(coords), 1);
    dist= max(sqrt(sum(diff.*diff,2)));
    
end