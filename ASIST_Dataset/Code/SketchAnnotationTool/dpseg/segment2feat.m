function feat = segment2feat(stroke, ind1, ind2, endwin)

    coords = stroke.coords(stroke.dppoints(ind1):stroke.dppoints(ind2),:);
    resampled_ind = stroke.resampled(:,3) >= stroke.dppoints(ind1) & stroke.resampled(:,3) <= stroke.dppoints(ind2);
    resampled = stroke.resampled(resampled_ind,:); 
    
    if (size(resampled,1) < 2)
        feat = zeros(1,23);
        return;
    end  
    
    leftind = find(stroke.dppoints(ind1) > [0;stroke.resampled(:,3)] & stroke.dppoints(ind1) <= [stroke.resampled(:,3);inf]);    
    left = stroke.resampled(max(1,leftind-endwin) : min(size(stroke.resampled,1), leftind+endwin), :);    
    rightind = find(stroke.dppoints(ind2) > [0;stroke.resampled(:,3)] & stroke.dppoints(ind2) <= [stroke.resampled(:,3);inf]); 
    right = stroke.resampled(max(1,rightind-endwin) : min(size(stroke.resampled,1), rightind+endwin), :);

%     plot_stroke(stroke), hold on,  plot(coords(:,1), -coords(:,2), '.'); plot(resampled(:,1), -resampled(:,2), 'r.');
%     plot(left(:,1), -left(:,2), 'cd'); plot(right(:,1), -right(:,2), 'cd')       
    
    feat = [];
    feat = [feat get_fit_results(resampled(:,[1 2]), ind2-ind1)];           % 1-15   
    feat = [feat get_length(coords)];                                       % 16
    feat = [feat (ind1==1 || ind2==length(stroke.dppoints))]; % is endpoint % 17
    feat = [feat get_max_diff(resampled(endwin+1:end-endwin-1, :))];        % 18-19
    feat = [feat get_max_diff(left)];                                       % 20-21
    feat = [feat get_max_diff(right)];                                      % 22-23
    
%     pause;
%     close;    

end


function feat = get_max_diff(resampled)

    npts = resampled(:,4); 
    angles = resampled(:,5);    
    metric = npts.*angles;
    metric(isnan(metric)) = -1; % dummy value;    
    [v i] = max(metric);    
    if (isempty(v) || v==-1)
        feat = [0 0];
    else
%         plot(resampled(i,1), -resampled(i,2), 'go');
%         display(sprintf('npts : % d, angle : %f, product : %f', npts(i), angles(i), metric(i)));
        feat = [npts(i) angles(i)];
    end
    
end


function feat = get_bezier_feats(coords, chordl)  

    if (length(coords) < 4)
        feat = [0 0];
    else
        ei= length(coords); 
        ibi=[1;ei]; %first and last point are taken as initial break points
        [p0mat,p1mat,p2mat,p3mat,fbi]=bzapproxu(coords, inf, ibi);
        f = BezierInterpCPMatSegVec(p0mat,p1mat,p2mat,p3mat,fbi);
%         diff = pdist2(coords, f, 'euclidean', 'Smallest',1);
        diff = pdist2(coords, f, 'euclidean');
        % Do?rulu?unu tekrar gözden geçir...
        feat = [sqrt(diff*diff')/chordl max(abs(diff))];        
    end
    
end



function feat = get_fit_results(coords, nbreaks)    
    feat = [];
    coords = pca_rotate(coords);   
    coords = coords + repmat(abs(min(coords)), size(coords,1), 1);
    bb = get_bounding_box(coords);
    chordl = get_length(coords);
    endl = get_distance(coords(1,:), coords(end,:));
    
    feat = [feat chordl/endl]; %1        
    feat = [feat bb.width/bb.height];    
    feat = [feat get_polyfit_feats(coords, 1, chordl)];
    feat = [feat get_polyfit_feats(coords, 2, chordl)];    
    feat = [feat get_spline_feats(coords,1, nbreaks, chordl)];
    feat = [feat get_spline_feats(coords,2, nbreaks, chordl)];
    feat = [feat get_bezier_feats(coords, chordl)];
    feat = [feat get_ellipse_feats(coords, chordl)];
    
end

function feat = get_ellipse_feats(coords, chordl)
    
    [x y] = deal(coords(:,1), coords(:,2));
    [e f] = fit_ellipse(x,y);  
    f = f';
    
    if (isempty(e) || ~isempty(e.status))
        feat = [0 0 0];
    else
%         diff = pdist2(coords, f, 'euclidean', 'Smallest',1);
        diff = pdist2(coords, f, 'euclidean');
        % Do?rulu?unu tekrar gözden geçir...        
        feat = [sqrt(diff*diff')/chordl ...
                max(abs(diff)) ...
                e.long_axis/e.short_axis];
    end
    
end


function feat = get_spline_feats(coords, degree, nbreaks, chordl)
    [x y] = deal(coords(:,1), coords(:,2)); 
    pp = splinefit(x,y,nbreaks,degree);       
    f =  ppval(pp,x);  
    diff = f-y;
    feat = [sqrt(diff'*diff)/chordl max(abs(diff))];
end

function feat = get_polyfit_feats(coords, degree, chordl)
    [x y] = deal(coords(:,1), coords(:,2)); 
    p = polyfit(x,y,degree);
    f = polyval(p,x);
    diff = f-y;
    feat = [sqrt(diff'*diff)/chordl max(abs(diff))];
end

function bounding_box = get_bounding_box(coords)    

    bounding_box.x1 = min(coords(:,1));
    bounding_box.y1 = min(coords(:,2));
    bounding_box.x2 = max(coords(:,1));
    bounding_box.y2 = max(coords(:,2));
    bounding_box.diagonal_length = get_distance([bounding_box.x1 bounding_box.y1],... 
                                                [bounding_box.x2 bounding_box.y2]);
    
    bounding_box.height = abs(bounding_box.y1 - bounding_box.y2);
    bounding_box.width = abs(bounding_box.x1 - bounding_box.x2);
    bounding_box.area = bounding_box.height*bounding_box.width;
end


function [coords rotmat_est] = pca_rotate(coords)

    x = coords(:,1);
    y = coords(:,2);

    %Remove the mean
    x = x(:)-mean(x(:));
    y = y(:)-mean(y(:));

    %Perform PCA
    c = cov(x, y);
    [eigenVectors, lambda] = eig(c);    
    [lambda,ind]=sort(diag(lambda));

    %Use the largest eigen value to extract the corresponding eigen vector
    indMax = ind(end);
    majorAxis = [ eigenVectors(1,indMax), eigenVectors(2,indMax) ];

    phi = atan2( majorAxis(2),majorAxis(1));
    phi = 2*pi - phi;
    rotmat_est = [cos(phi), sin(phi); -sin(phi), cos(phi)];
    unrotated = inv(rotmat_est)*[x y]';    
    x = unrotated(1,:);
    y = unrotated(2,:);  
    coords = [x' y'];
    
end


function dist = pdist2(coords,f, metric)
    D = squareform(pdist([coords;f], metric));
    N = size(coords,1);
    D = D(N+1:end, 1:N);
    dist = min(D,[],1);
end