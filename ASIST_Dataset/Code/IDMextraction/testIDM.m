function testIDM( coords, idmvec )

figure; hold on;
for pnt = 1 : length(coords)
   plot( coords(pnt,1), coords(pnt,2), 'r*' );
end

visualizeIDM(idmvec);


end

