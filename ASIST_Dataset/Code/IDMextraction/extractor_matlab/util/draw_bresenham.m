function im = draw_bresenham(im, coords, pixelval)
    for i=1:length(pixelval)
        c = [coords(i,:);coords(i+1,:)];
        if (pixelval(i)>0)              
            [x y] = bresenham(c(1,1),c(1,2),c(2,1),c(2,2));
            for j=1:length(x)
                try
                    if (im(y(j), x(j)) < pixelval(i))
                        im(y(j), x(j)) = pixelval(i);
                    end
                catch err
                    disp(strcat('Got youuuuu-----------------------------------',num2str(y(j)),num2str(x(j))));
                end
                
            end
        end
    end        
end