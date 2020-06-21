function distance = segments_dist_3d ( p1, p2, q1, q2 )

%*****************************************************************************80
%
%% SEGMENTS_DIST_3D computes the distance between two line segments in 3D.
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
%    Input, real P1(3), P2(3), the endpoints of the first segment.
%
%    Input, real Q1(3), Q2(3), the endpoints of the second segment.
%
%    Output, real DISTANCE, the distance between the line segments.
%
  dim_num = 3;
%
%  Find the nearest points on line 2 to the endpoints of line 1.
%
  [ pn1, d1, t1 ] = segment_point_near_3d ( q1, q2, p1 );
  [ pn2, d2, t2 ] = segment_point_near_3d ( q1, q2, p2 );

  if ( t1 == t2 )
    distance = segment_point_dist_3d ( p1, p2, pn1 );
    return
  end

  pm(1:dim_num) = 0.5 * ( pn1(1:dim_num) + pn2(1:dim_num) );
%
%  On line 2, over the interval between the points nearest to line 1,
%  the square of the distance of any point to line 1 is a quadratic function.
%  Evaluate it at three points, and seek its local minimum.
%
  dl = segment_point_dist_3d ( p1, p2, pn1 );
  dm = segment_point_dist_3d ( p1, p2, pm );
  dr = segment_point_dist_3d ( p1, p2, pn2 );

  tl = 0.0;
  tm = 0.5;
  tr = 1.0;

  dl = dl * dl;
  dm = dm * dm;
  dr = dr * dr;

  [ tmin, distance ] = minquad ( tl, dl, tm, dm, tr, dr );

  distance = sqrt ( distance );

  return
end
