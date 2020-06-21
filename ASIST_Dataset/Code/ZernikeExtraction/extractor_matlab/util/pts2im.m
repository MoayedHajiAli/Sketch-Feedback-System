function im = pts2im(coords_per_stroke, imsize, pixelval)
     
     center = [0 0];
     scale = imsize./(2.5*2);  
     im = zeros(imsize);
     for i=1:length(coords_per_stroke)
        transformed = transform(coords_per_stroke{i}, center, scale, imsize);
        if (~exist('pixelval', 'var'))
            im = draw_bresenham(im, transformed, ones(1,size(coords_per_stroke{i},1)-1)); 
        else
            im = draw_bresenham(im, transformed, pixelval{i}); 
        end
            
     end

end

