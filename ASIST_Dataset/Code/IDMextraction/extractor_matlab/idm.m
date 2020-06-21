function feat = idm(pts, resample_interval, sigma, hsize)

    gridwidth = 12; 
    gridheight = 12;    
    [coords coords_per_stroke] = resampleExtractor(pts, resample_interval, false);     
    % normalize 
    [coords coords_per_stroke] = normalize(coords_per_stroke);
    theta = coords2angles(coords_per_stroke);
    
    featim1 = featim(theta, coords_per_stroke, gridwidth, gridheight, 0, false, sigma, hsize);
    featim2 = featim(theta, coords_per_stroke, gridwidth, gridheight, 45, false, sigma, hsize);
    featim3 = featim(theta, coords_per_stroke, gridwidth, gridheight, 90, false, sigma, hsize);
    featim4 = featim(theta, coords_per_stroke, gridwidth, gridheight, 135, false, sigma, hsize);
    featim5 = featim(theta, coords_per_stroke, gridwidth, gridheight, nan, true, sigma, hsize);
    
    
%     subplot(2,3,1), imshow(featim1);
%     subplot(2,3,2), imshow(featim2);
%     subplot(2,3,3), imshow(featim3);
%     subplot(2,3,4), imshow(featim4);   
%     subplot(2,3,5), imshow(featim5);  
    
    feat = [featim1(:);featim2(:);featim3(:);featim4(:);featim5(:)]';
    
    
end

function theta = coords2angles(coords_per_stroke)
    for i=1:length(coords_per_stroke)
        scoords = coords_per_stroke{i};
        delta = [scoords(2:end,:); nan nan] - scoords;
        delta(end,:) = [];
        theta{i} = [atan3(delta(:,2),delta(:,1));nan];
    end
end

function im = featim(theta, coords_per_stroke, gridwidth, gridheight, current_angle, endpts, sigma, hsize)

    center = [0 0];
    % downsample later
    imsize = [gridwidth*2, gridheight*2];
    
    % At this point, width and height are scaled so that current stdev is 1.0 in both directions.
    % Since the grid needs to span 2.5 stdev, we need to scale the current symbol with 1/2.5
    % and then 0.5 (because origin is the center of the image).    
    scale = imsize./(2.5*2); % imsize./(2*2);
    
    current_angle2 = mod(current_angle + 180, 360);
    im = zeros(imsize);
    if (~endpts)   
        angledist1 = get_angle_distance(theta, current_angle);
        angledist2 = get_angle_distance(theta, current_angle2);
        for i=1:length(angledist1)
            mindist = min([angledist1{i} angledist2{i}],[],2); 
            pixelval{i} = pixval(mindist);
            pixelval{i}(end) = []; % last is nan
        end
        im = pts2im(coords_per_stroke, imsize, pixelval);
    else
        for i=1:length(coords_per_stroke)
            ends = coords_per_stroke{i}([1,end],:);
            transformed = transform(ends, center, scale, imsize);
            im(transformed(:,2), transformed(:,1)) = 1.0;
        end
    end
    im = smoothim(im, sigma, hsize); 
    im = downsample(im);    
    
end

function downim = downsample(im)
    downim = zeros(size(im)/2);
    for i=1:size(im,1)/2
        for j=1:size(im,2)/2
            neighborhood = im([i*2-1 i*2], [j*2-1 j*2]);
            downim(i,j) = max(max(neighborhood));
        end
    end
end

 
function dist = pixval(dist)
    anglethresh = 45;
    valid_indices = dist<=anglethresh;
    dist(valid_indices) = 1-(dist(valid_indices)/anglethresh); 
    dist(~valid_indices) = 0;  
end


function angledist = get_angle_distance(theta, current_angle)
    for i=1:length(theta)
        diff = theta{i} - current_angle;
        diff(diff>=180) = diff(diff>=180)-360;
        diff(diff<=-180) = diff(diff<=-180)+360;
        angledist{i} = abs(diff);        
    end    
end