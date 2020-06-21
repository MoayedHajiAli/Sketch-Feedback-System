function [ nline, line ] = sphere_imp_gridlines_3d ( maxline, nlat, nlong )

%*****************************************************************************80
%
%% SPHERE_IMP_GRIDLINES_3D produces "grid lines" on an implicit sphere in 3D.
%
%  Discussion:
%
%    The point numbering system is the same used in SPHERE_IMP_GRIDPOINTS_3D,
%    and that routine may be used to compute the coordinates of the points.
%
%    An implicit sphere in 3D satisfies the equation:
%
%      sum ( ( P(1:DIM_NUM) - CENTER(1:DIM_NUM) )**2 ) = R**2
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    23 May 2005
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer MAXLINE, the maximum number of gridlines.
%
%    Input, integer NLAT, NLONG, the number of latitude and longitude
%    lines to draw.  The latitudes do not include the North and South
%    poles, which will be included automatically, so NLAT = 5, for instance,
%    will result in points along 7 lines of latitude.
%
%    Output, integer NLINE, the number of grid lines.
%
%    Output, integer LINE(2,MAXLINE), contains pairs of point indices for
%    line segments that make up the grid.
%
  nline = 0;
%
%  "Vertical" lines.
%
  for j = 0 : nlong - 1

    old = 1;
    new = j + 2;

    if ( nline < maxline )
      nline = nline + 1;
      line(1:2,nline) = [ old, new ]';
    end

    for i = 1 : nlat - 1

      old = new;
      new = old + nlong;

      if ( nline < maxline )
        nline = nline + 1;
        line(1:2,nline) = [ old, new ]';
      end

    end

    old = new;

    if ( nline < maxline )
      nline = nline + 1;
      line(1:2,nline) = [ old, 1 + nlat * nlong + 1 ]';
    end

  end
%
%  "Horizontal" lines.
%
  for i = 1 : nlat

    new = 1 + ( i - 1 ) * nlong + 1;

    for j = 0 : nlong - 2
      old = new;
      new = old + 1;
      if ( nline < maxline )
        nline = nline + 1;
        line(1:2,nline) = [ old, new ]';
      end
    end

    old = new;
    new = 1 + ( i - 1 ) * nlong + 1;
    if ( nline < maxline )
      nline = nline + 1;
      line(1:2,nline) = [ old, new ]';
    end

  end
%
%  "Diagonal" lines.
%
  for j = 0 : nlong - 1

    old = 1;
    new = j + 2;
    newcol = j;

    for i = 1 : nlat - 1

      old = new;
      new = old + nlong + 1;

      newcol = newcol + 1;
      if ( nlong - 1 < newcol )
        newcol = 0;
        new = new - nlong;
      end

      if ( nline < maxline )
        nline = nline + 1;
        line(1:2,nline) = [ old, new ]';
      end

    end

  end

  return
end

