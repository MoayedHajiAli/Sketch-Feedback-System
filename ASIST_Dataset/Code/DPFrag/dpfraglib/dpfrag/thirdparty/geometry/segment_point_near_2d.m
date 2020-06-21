function [ pn, dist, t ] = segment_point_near_2d ( p1, p2, p )

%*****************************************************************************80
%
%% SEGMENT_POINT_NEAR_2D finds the line segment point nearest a point in 2D.
%
%  Discussion:
%
%    A line segment is the finite portion of a line that lies between
%    two points.
%
%    The nearest point will satisfy the condition
%
%      PN = (1-T) * P1 + T * P2.
%
%    T will always be between 0 and 1.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    03 May 2006
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real P1(2), P2(2), the endpoints of the line segment.
%
%    Input, real P(2), the point whose nearest neighbor
%    on the line segment is to be determined.
%
%    Output, real PN(2), the point on the line segment which is
%    nearest the point (X,Y).
%
%    Output, real DIST, the distance from the point to the
%    nearest point on the line segment.
%
%    Output, real T, the relative position of the point (XN,YN)
%    to the points (X1,Y1) and (X2,Y2).
%
  dim_num = 2;
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

  dist = sqrt ( sum ( ( pn(1:dim_num) - p(1:dim_num) ).^2 ) );

  return
end
