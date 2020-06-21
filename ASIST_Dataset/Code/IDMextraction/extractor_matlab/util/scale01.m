function im = scale01(im)
    im = im-min(min(im));
    im = im/max(max(im));
end

