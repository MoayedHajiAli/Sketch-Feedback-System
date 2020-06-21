function params = get_line_params(coords)

    if (size(coords,2) ~= 2)
        type = -1;
        return;
    end    

    [x y] = deal(coords(:,1), coords(:,2)); 
    p = polyfit(x,y,1);
    f = polyval(p,x);
    newcoords = [x f];
    params = [newcoords(1,:); newcoords(end,:)];
    
    hold on;
    plot(params(:,1), params(:,2), 'g');

end

