function prec = dpseg_train(trs, params)

    trs = douglas_peucker(trs, params.douglasthresh);    
    trs = resample(trs, params.resample_interval, params.anglemeasure);
    [feat lab] = extract_feats(trs, params.endwin);      
    prec = classifier(feat, lab, 'train');
    
end