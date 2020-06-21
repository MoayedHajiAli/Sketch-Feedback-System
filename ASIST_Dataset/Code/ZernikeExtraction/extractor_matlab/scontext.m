function feat = scontext(pts, nrbin, ntheta, maxradius)

    coords = resampleExtractor(pts, 50, false);
    bb = get_bounding_box(cell2mat({pts.coords}'));
    if (bb.width > bb.height)
        dscale = 100/bb.width;
    else
        dscale = 100/bb.height;
    end
    coords = coords - repmat([bb.x1, bb.y1], size(coords,1), 1);
    coords = coords*dscale;
    
    newwidth = bb.width*dscale;
    newheight = bb.height*dscale;
       
    feat1 = getsc(coords, nrbin, ntheta, maxradius,[0 0]);    
    feat1 = prune(feat1,0,90);
    feat2 = getsc(coords, nrbin, ntheta, maxradius,[newwidth 0]);
    feat2 = prune(feat2,90,180);
    feat3 = getsc(coords, nrbin, ntheta, maxradius,[newwidth newheight]); 
    feat3 = prune(feat3,180,270);    
    feat4 = getsc(coords, nrbin, ntheta, maxradius,[0 newheight]);
    feat4 = prune(feat4,270,360);  
    feat5 = getsc(coords, nrbin, ntheta, maxradius,[newwidth/2 newheight/2]);    

    feat = [feat1(:);feat2(:);feat3(:);feat4(:);feat5(:)];
    feat = feat';
    
    
end

function feat = prune(feat, startt, endt)
    startbin = ceil((startt/360*size(feat,1))+ 10^-10);
    endbin = ceil((endt/360)*size(feat,1));
    feat = feat(startbin:endbin,:);
end


function feat = getsc(coords, nrbin, ntheta, maxradius, loc)

    delta = coords - repmat(loc, size(coords,1),1);
    dist = sqrt(sum(delta.*delta,2));
    
    delta = delta(dist<=maxradius,:);
    coords = coords(dist<=maxradius,:);
    dist = dist(dist<=maxradius,:);   
    theta = atan3(delta(:,2), delta(:,1));
    
    dist = dist+eps;
    thetabin = ceil(ntheta*(theta/360));
    rbin = ceil(nrbin*(log(dist)/log(maxradius)));

%     figure,
%     plot(coords(:,1), coords(:,2), '.')  
%     hold on, plot(loc(1), loc(2), '*')
%     for i=1:nrbin
%        circle(loc, exp((i/nrbin)*log(maxradius)),1000,'--'); 
%     end    
%     
%     pause;
%     close all;
    
    for i=1:ntheta
        for j=1:nrbin
            feat(i,j) = sum(thetabin == i & rbin == j);
        end
    end    

end