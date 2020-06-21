function [coords coords_per_stroke] = resampleExtractor(pts, ratio, to_plot)
    % ratio: typically 50
    
    maxdist = get_outlier(cell2mat({pts.coords}'));
    maxdist = maxdist*1.01;
    interval = maxdist/ratio;
    nstrokes = length(pts);
    coords = [];
    for s = 1:nstrokes
        new_coords = [];
        spts = pts(s);
        prev = spts.coords(1,:);
        new_coords(end+1,:) = prev;
        for i=2:length(spts.coords) %Was spts.time
            while (get_distance(spts.coords(i,:), prev) > interval)
                c = prev;
                angle = atan2(spts.coords(i,2) - c(2), spts.coords(i,1) - c(1));
                prev = [c(1) + cos(angle)*interval, ...
                        c(2) + sin(angle)*interval];
                new_coords = [new_coords; prev];     
            end
        end
        coords_per_stroke{s} = new_coords;
        coords = [coords; new_coords];
    end
    
    if (to_plot)
        plot(coords(:,1), -coords(:,2), '.');
        axis equal;
    end
    
end
