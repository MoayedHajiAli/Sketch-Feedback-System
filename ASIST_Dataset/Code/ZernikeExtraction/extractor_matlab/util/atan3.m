function theta = atan3(dy, dx)

    theta = atan2(dy, dx);
    theta = mod(theta,2*pi)*180/pi;

end

