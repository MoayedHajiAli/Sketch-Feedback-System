function dist = quad_point_dist_2d ( q, p )

%*****************************************************************************80
%
%% QUAD_POINT_DIST_2D: distance ( quadrilateral, point ) in 2D.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    18 January 2007
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real Q(2,4), the quadrilateral vertices.
%
%    Input, real P(2), the point to be checked.
%
%    Output, real DIST, the distance from the point to the
%    quadrilateral.
%
  dim_num = 2;
  nside = 4;
%
%  Find the distance to each of the line segments.
%
  dist = Inf;

  for j = 1 : nside

    jp1 = i4_wrap ( j+1, 1, nside );

    dist2 = segment_point_dist_2d ( q(1:dim_num,j)', q(1:dim_num,jp1)', ...
      p(1:dim_num) );

    if ( dist2 < dist )
      dist = dist2;
    end

  end

  return
end
