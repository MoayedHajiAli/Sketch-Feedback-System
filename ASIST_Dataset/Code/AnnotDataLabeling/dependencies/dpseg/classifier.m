function [r1 r2] = classifier(feat, arg, type, featset)
    switch type
        
        case 'train'
            if (exist('featset', 'var'))
                feat = feat(:, featset);
            end            
            [r1 r2] = train_recognizer(feat, arg);
            if (exist('featset', 'var'))
                r1.featset = featset;
            else
                r1.featset = 1:size(feat,2);
            end            
            
        case 'predict'
            [r1 r2] =  predict(feat(:,arg.featset), arg);             
    end
end


function [predict_label prob_estimates] = predict(feat, recognizer)

    feat = scale_feats(feat, recognizer.mins, recognizer.maxs);
    [predict_label, accuracy, prob_estimates] = svmpredict(zeros(size(feat,1),1), feat, recognizer.W, '-b 1');
    
end

function [recognizer bestcv] = train_recognizer(feat, label)

    [label i] = sort(label);
    feat = feat(i,:);
  
    recognizer.maxs = max(feat,[],1);
    recognizer.mins = min(feat,[],1);
    
    feat = scale_feats(feat, recognizer.mins, recognizer.maxs);

    bestcv = 0;
    for log2c = -1.1:3.1,
        for log2g = -4.1:1.1
            cmd = ['-t 2 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
            cv = svmtrain(label, feat, cmd);
            if (cv >= bestcv),
                bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
%                 fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
            end
        end
    end

    cmd = ['-t 2 -c ', num2str(bestc), ' -g ', num2str(bestg) ' -b 1'];
    recognizer.W = svmtrain(label, feat, cmd);
    fprintf('best c=%g, g=%g, rate=%g\n', bestc, bestg, bestcv);

end


function feat = scale_feats(feat, mins, maxs)
    if (isempty(feat))
        return;
    end
    nrows = size(feat,1);
    ncols = size(feat,2);    
    % scale data    
    t1 = repmat(mins,nrows,1);
    t2 = spdiags(1./(maxs - mins)',0, ncols , ncols);
    feat = (feat - t1) * t2;

end



