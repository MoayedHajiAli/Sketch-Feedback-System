function printCoords( coords )

figure; hold on; grid on; grid minor;
for i = 1 : length(coords)
    plot( coords(i,1), coords(i,2), 'b*' );
end

end

