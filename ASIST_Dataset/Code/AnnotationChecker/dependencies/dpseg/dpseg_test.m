function [aons fragmented_tes] = dpseg_test(tes, precs, params)    
    tes = douglas_peucker(tes, params.douglasthresh);
    tes = resample(tes, params.resample_interval, params.anglemeasure); 
    for i=1:length(precs)
        fragmented_tes{i} = fragstrokes(tes, precs(i), params);
        aons(i) = evaluate(fragmented_tes{i}, 'dpseg', params.eval_tolerance);        
    end    
end

function strokes = fragstrokes(strokes, prec, params)
    strokes().dpseg = [];
    for i=1:length(strokes)
        strokes(i) = fragstroke(strokes(i), prec, params);
        display(sprintf('test stroke fragmented : %d', i));
    end
end