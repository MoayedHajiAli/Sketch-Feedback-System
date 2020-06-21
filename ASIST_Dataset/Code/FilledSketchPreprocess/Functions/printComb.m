function printComb( combs, a )

figure; hold on;
for pnt = 1 : length(combs{a}.coords)
   plot( combs{a}.coords(pnt,1), combs{a}.coords(pnt,2), 'r*' );
end

end

