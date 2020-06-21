function distance = segment_point_dist_3d ( p1, p2, p )

%*****************************************************************************80
%
%% SEGMENT_POINT_DIST_3D: distance ( line segment, point ) in 3D.
%
%  Discussion:
%
%    A line segment is the finite portion of a line that lies between
%    two points.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    20 July 2006
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real P1(3), P2(3), the endpoints of the line segment.
%
%    Input, real P(3), the point whose nearest neighbor on the line
%    segment is to be determined.
%
%    Output, real DISTANCE, the distance from the point to the line segment.
%
  dim_num = 3;
%
%  If the line segment is actually a point, then the answer is easy.
%
  if ( p1(1:dim_num) == p2(1:dim_num) )

    t = 0.0;

  else

    bot = sum ( ( p2(1:dim_num) - p1(1:dim_num) ).^2 );

    t = ( p(1:dim_num) - p1(1:dim_num) ) ...
      * ( p2(1:dim_num) - p1(1:dim_num) )' / bot;

    t = max ( t, 0.0 );
    t = min ( t, 1.0 );

  end

  pn(1:dim_num) = p1(1:dim_num) + t * ( p2(1:dim_num) - p1(1:dim_num) );

  distance = sqrt ( sum ( ( pn(1:dim_num) - p(1:dim_num) ).^2 ) );

  return
end
