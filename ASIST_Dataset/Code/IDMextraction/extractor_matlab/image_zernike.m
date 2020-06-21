function feat = image_zernike(pts, order)

    nstrokes = length(pts);
    ncoords = [];
    for i=1:nstrokes
        scoords = round(pts(i).coords);
        ncoords = [ncoords; interpolate(scoords)];
    end
    feat = zernike(ncoords, order);
    
end


function ncoords = interpolate(coords)
    delta = [coords(2:end,:); nan nan] - coords;
    delta(end,:) = [];
    dist = sqrt(sum(delta.*delta,2));
    ncoords = [];
    for i=1:size(coords,1)-1
        ncoords(end+1,:) = coords(i,:);
        for k=1:dist(i)-1
            ncoords(end+1,:) = coords(i,:) + delta(i,:)* (k/dist(i));
        end
    end
    ncoords(end+1,:) = coords(end,:);
end


