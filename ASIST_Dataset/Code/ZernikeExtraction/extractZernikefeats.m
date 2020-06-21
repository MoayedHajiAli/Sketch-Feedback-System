function [ zernikeVector ] = extractZernikefeats( sketchData )
% sketchData:
% -- coords
% -- strokeIDs

% Zernike Parameters
order = 15;

%% Extract feature vector
zernikeVector = zernike(sketchData.coords, order);

end

