function plotComb( comb, pref )

    figure; hold on;
    for pnt = 1 : size(comb.coords,1)
       coords = comb.coords(pnt,:);
       plot( coords(1), -coords(2), pref );
    end

end

