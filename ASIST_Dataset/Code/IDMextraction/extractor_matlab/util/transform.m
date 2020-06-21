function transformed = transform(scoords, center, scale, imsize)
    n = size(scoords,1);
    transformed = repmat(imsize/2,n,1) + (repmat(scale,n,1).*(scoords-repmat(center,n,1)));
    transformed = floor(transformed);
    for i=1:size(transformed,2)
        for j=1:size(transformed,1)
            if(transformed(j,i)>imsize(1,1))
                transformed(j,i)=imsize(1,1);
            end
            if(transformed(j,i)<1)
                transformed(j,i)=1;
            end
        end
    end
end
