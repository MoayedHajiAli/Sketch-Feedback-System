function plot_stroke(arg, type)    

    if (~exist('type', 'var'))
        type = {'stroke', 'primitives', 'corners', 'ends', 'dppoints', 'dpseg', 'istraw', 'resampled'};
    end
    
    hold on;
    for i = 1:length(type)
        switch type{i}  
            case 'stroke'
                plot_strokes(arg);
            case 'primitives'
                plot_primitives(arg);   
            case 'corners'
                plot_corners(arg);
            case 'ends'   
                plot_end_points(arg); 
            case 'resampled'
                plot_resampled(arg);
            case 'dppoints'
                plot_dppoints(arg);
            case 'dpseg'
                plot_dpseg(arg);
            case 'istraw'
                plot_istraw(arg);                
        end
    end
    hold off;
    axis equal;
end


function plot_dpseg(stroke)
    if (isfield(stroke, 'dpseg'))
        coords = stroke.coords(stroke.dpseg,:);
        plot(coords(:,1), -coords(:,2), 'gd', 'MarkerSize',14);
    end
end


function plot_istraw(stroke)
    if (isfield(stroke, 'istraw'))
        coords = stroke.coords(stroke.istraw,:);
        plot(coords(:,1), -coords(:,2), 'mo', 'MarkerSize',10);
    end
end

function plot_resampled(stroke)
    if (isfield(stroke, 'resampled')) 
        plot(stroke.resampled(:,1), -stroke.resampled(:,2), 'x'); 
    end
end

function plot_dppoints(stroke)
    if (isfield(stroke, 'dppoints')) 
        coords = stroke.coords(stroke.dppoints,:);
        plot(coords(:,1), -coords(:,2), 'kd');        
    end
end

function plot_end_points(stroke)
   plot(stroke.coords([1 end], 1), -stroke.coords([1 end], 2), 'r*'); 
end

function plot_corners(stroke)
    plot(stroke.coords(stroke.corners, 1), -stroke.coords(stroke.corners, 2), 'r*');
end


function plot_strokes(stroke)    
    plot(stroke.coords(:,1), -stroke.coords(:,2), 'Color', [0.9 0.9 0.9]);
    plot(stroke.coords(:,1), -stroke.coords(:,2), '.', 'Color', [0.3 0.3 0.3]);
end


function plot_primitives(stroke)    
    for i=1:length(stroke.nprims)
        indices = stroke.primids == stroke.nprims(i);
        primtype = unique(stroke.primtypes(indices));
        coords = stroke.coords(indices, :);
        plot(coords(:,1), -coords(:,2), '--', 'Color', get_primitive_info(primtype, 'color'));
    end
end



