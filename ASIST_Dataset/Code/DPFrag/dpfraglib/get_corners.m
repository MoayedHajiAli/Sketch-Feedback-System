function indices = get_corners(xmlfile)

    strokes = read_sketch(xmlfile);
    load env_annot_thres3.mat;
    
%     params = getparams('deneme');
%     [trs tes trf tef] = get_strokes(params.setdir, params.extension, params.exceptions, .8);
%     params = get_bestparams(trs, params);
%     prec = dpseg_train(trs, params);
    for i=1:length(strokes)        
        current = strokes(i);
        current = douglas_peucker(current, 1);
        current = resample(current, 1, params.anglemeasure);     
        current = fragstroke(current, prec, params);
        indices{i} = current.dpseg; %cell yaptým                  
    end
    
end

