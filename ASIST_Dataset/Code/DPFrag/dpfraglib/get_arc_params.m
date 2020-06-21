function params = get_arc_params( coords )

% ellipse_t structure : 
% -------------------------
% a           - sub axis (radius) of the X axis of the non-tilt ellipse
% b           - sub axis (radius) of the Y axis of the non-tilt ellipse
% phi         - orientation in radians of the ellipse (tilt)
% X0          - center at the X axis of the non-tilt ellipse
% Y0          - center at the Y axis of the non-tilt ellipse
% X0_in       - center at the X axis of the tilted ellipse
% Y0_in       - center at the Y axis of the tilted ellipse
% long_axis   - size of the long axis of the ellipse
% short_axis  - size of the short axis of the ellipse
% status      - status of detection of an ellipse

    [x y] = deal(coords(:,1), coords(:,2)); 
    [ellipse_t rotated_ellipse] = fit_ellipse(x,y);
   
    % ?????? bu length 0 oldugu durumlarda rotated_ellipse bulunamýyor
    if length(rotated_ellipse)>0
    hold on;
    plot(rotated_ellipse(1,:),rotated_ellipse(2,:),'g' );
    end
    
    params = [ellipse_t.long_axis, ...
              ellipse_t.short_axis, ...
              ellipse_t.X0_in,...
              ellipse_t.Y0_in, ...
              ellipse_t.phi];
    
   
end

