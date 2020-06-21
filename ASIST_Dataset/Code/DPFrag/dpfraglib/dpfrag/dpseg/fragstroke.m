function stroke = fragstroke(strokes, prec, params)
    stroke = fragstroke_start2end(strokes, prec, params);
end

function s = fragstroke_end2start(s, prec, params)
    % from end to start
    n = length(s.dppoints);
    g = cell(1,n);
    g{1}.cost = 0;
    g{1}.segment = [];
    for i=2:n
%         display('-------------------');
        g{i}.cost = inf;
        for j=1:i-1
            f = segment2feat(s,j,i, params.endwin);          
            [lab prob] = classifier(f, prec, 'predict');
            pprim = 1-prob(end);          
            cost = -log(pprim) +  g{j}.cost;
%             display(sprintf('g(%d) + C{%d,%d} = %f', j,j,i, cost));
            if (g{i}.cost > cost)                
                g{i}.cost = cost;
                g{i}.segment = [j i];
            end                            
        end
    end   
    
    ind = n;
    c = s.dppoints;
    s.dpseg = c(g{ind}.segment(2));
    while (ind~=1)
        s.dpseg = [c(g{ind}.segment(1)) s.dpseg];
        ind = g{ind}.segment(1);
    end

    
%     plot_stroke(s);
%     pause;
%     close;

    
    
end





function s = fragstroke_start2end(s, prec, params)


    % from end to start
    n = length(s.dppoints);
    g = cell(1,n);
    g{end}.cost = 0;
    g{end}.segment = [];
    for i=n-1:-1:1
%         display('-------------------');
        g{i}.cost = inf;
        for j=i+1:min(n,n)
            f = segment2feat(s,i,j, params.endwin);
            [lab prob] = classifier(f, prec, 'predict');
            pprim = 1-prob(end);          
            cost = 1/pprim +  g{j}.cost;
%             display(sprintf('C{%d,%d} + g(%d) = %f', i,j,j, cost));
            if (g{i}.cost > cost)                
                g{i}.cost = cost;
                g{i}.segment = [i j];
            end                            
        end
    end   
    
    ind = 1;
    c = s.dppoints;
    s.dpseg = c(g{ind}.segment(1));
    while (ind~=n)
        s.dpseg = [s.dpseg c(g{ind}.segment(end))];
        ind = g{ind}.segment(end);
    end
    
%     plot_stroke(s);
%     pause;
%     close;

    
    
end