function im = smoothim(im, sigma, hsize)
    gaussfilter = fspecial('gaussian', hsize, sigma);
    im = imfilter(im, gaussfilter, 'same');
    temp=max(max(im));
    if(temp~=0)
        im = im/temp;
    end
end

