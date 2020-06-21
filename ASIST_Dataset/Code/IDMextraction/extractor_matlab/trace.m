function tracef = trace(pts, nbin, ntheta, sigma, hsize, colfun, rowfun)
   
    [coords coords_per_stroke] = resampleExtractor(pts, 10, false);
    [coords coords_per_stroke] = normalize(coords_per_stroke);
    imsize = [100 100];
    im = pts2im(coords_per_stroke, imsize);
    im = smoothim(im, sigma, hsize);    
    tracef = im2tracef(im, nbin, ntheta, colfun,rowfun); 
    
end


function tracef = im2tracef(im, nbin, ntheta, colfun,rowfun)
    for t=1:ntheta
        angle = (t-1)*180/ntheta;
        im2 = imrotate(im,angle);        
        dscale = nbin/size(im2,2);
        nscaledw = floor(nbin);
        nscaledh = floor(size(im2,1)*dscale); 
        im2 = imresize(im2, [nscaledh nscaledw]);  
        im2 = scale01(im2);
        % calculate f1(x) of columns
        C = num2cell(im2,1);
        traceim(:,t) = cellfun(colfun,C);        
    end
    % calculate f2(x) of rows of traceim
    C = num2cell(traceim,2);
    tracef = cellfun(rowfun,C);    
end

