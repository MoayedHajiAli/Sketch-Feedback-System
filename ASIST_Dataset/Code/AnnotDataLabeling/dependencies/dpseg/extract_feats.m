function [feat lab] = extract_feats(strokes, endwin)
    feat=[]; lab=[];
    for i=1:length(strokes)
        display(sprintf('Extracting features : %d', i));
        [f l] = stroke2feats(strokes(i), endwin);
        feat = [feat; f];
        lab = [lab; l];       
    end
	unknown = (lab == 4);
	feat(unknown, :) = [];
	lab(unknown) = [];     
end

function [feat lab] = stroke2feats(s, endwin)
    feat=[]; lab=[]; 
    c = s.dppoints;
    for i=1:length(c)-1
        for j= i+1:min(length(c),length(c))
            l = getlab(s, c(i), c(j));
            feat = [feat; segment2feat(s,i,j,endwin)];
            lab = [lab; l];  
        end
    end
end


function l = getlab(stroke, i0, i1)
    
    % l = 1 -> line
    % l = 2 -> arc
    % l = 3 -> composite
    % l = 4 -> unknown
    
    uniqueprimids = unique(stroke.primids);
    stroke_dist = [];
    for i=1:length(uniqueprimids)
        stroke_dist = [stroke_dist get_length(stroke.coords(stroke.primids == uniqueprimids(i),:))]; 
    end
    
    inseg_prims = stroke.primids(i0:i1);
    inseg_dist = [];
    for i = 1:length(uniqueprimids)
        temp = [stroke.coords(i0:i1,:) stroke.primids(i0:i1)];
        inseg_dist = [inseg_dist get_length(temp(temp(:,3) == uniqueprimids(i), [1 2]))];
    end    
    
    hist_ratio = inseg_dist ./ stroke_dist;
    thresh0 = hist_ratio > 0.95;
    thresh1 = hist_ratio < 0.05;    
    if (sum(thresh0) == 1 && sum(thresh1) == length(uniqueprimids)-1)
        
        type = stroke.primtypes(stroke.primids == uniqueprimids(thresh0));   
        type = type(1);
        if (type == get_primitive_info('line', 'label'))
            l = 1;
        elseif (type == get_primitive_info('bezier', 'label'))
            l = 2;
        else
            l = 4;
        end  
        
    elseif (sum(thresh0) == 0)
        l=4;
    else
        l=3;
    end      
    
end
