function type = get_primitive_type(coords, times)  
    
    % type = 1 if line
    % type = 2 if elliptic arc 
    
    load env;
    stroke.coords = coords;
    stroke.times = times;
    stroke.dppoints = [1;length(times)];    
    stroke = resample(stroke, params.resample_interval, params.anglemeasure);
    feat = segment2feat(stroke, 1,2, params.endwin);
    [lab prob] = classifier(feat, prec, 'predict');    
    [v type] = max(prob(1:2));
    
    
end