function strokes = douglas_peucker(strokes, tol)
    strokes().dppoints = [];
    for i=1:length(strokes)
        [ps,ix] = dpsimplify(strokes(i).coords, tol);
        strokes(i).dppoints = ix;
    end
end
