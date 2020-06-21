function feat = resample_zernike(pts, order)
    coords = resampleExtractor(pts, 50, false);
    feat = zernike(coords, order);
end

