function bestparams = get_bestparams(trs, params)
   
    bestparams = params;
    best_accuracy = 0;
    
    trs = douglas_peucker(trs, params.douglasthresh); 
    for i=1:length(params.resample_interval)
        for j=1:length(params.anglemeasure)
            trs = resample(trs, params.resample_interval(i), params.anglemeasure(j));
            for k=1:length(params.endwin)
                [feat lab] = extract_feats(trs, params.endwin(k));
                [prec accuracy] = classifier(feat, lab, 'train');
                if (best_accuracy < accuracy)
                    bestparams.resample_interval = params.resample_interval(i);
                    bestparams.anglemeasure = params.anglemeasure(j);
                    bestparams.endwin = params.endwin(k);
                    best_accuracy = accuracy;
                end
            end
        end
    end
    
end