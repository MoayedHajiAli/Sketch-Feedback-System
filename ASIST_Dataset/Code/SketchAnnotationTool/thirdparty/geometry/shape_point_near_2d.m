function [ pn, dist ] = shape_point_near_2d ( center, p1, nside, p, pn, dist )

%*****************************************************************************80
%
%% SHAPE_POINT_NEAR_2D: nearest point ( regular shape, point ) in 2D.
%
%  Discussion:
%
%    The "regular shape" is assumed to be an equilateral and equiangular
%    polygon, such as the standard square, pentagon, hexagon, and so on.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    22 May 2005
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real CENTER(2), the center of the shape.
%
%    Input, real P1(2), the first vertex of the shape.
%
%    Input, integer NSIDE, the number of sides in the shape.
%
%    Input, real P(2), the point to be checked.
%
%    Output, real PN(2), the point on the shape that is nearest
%    to the given point.
%
%    Output, real DIST, the distance between the points.
%
  dim_num = 2;
%
%  Determine the angle subtended by a single side.
%
  sector_angle = 360.0 / nside;
%
%  How long is the half-diagonal?
%
  radius = sqrt ( sum ( ( p1(1:dim_num) - center(1:dim_num) ).^2 ) );
%
%  If the radius is zero, then the shape is a point and the computation is easy.
%
  if ( radius == 0.0 )
    pn(1:dim_num) = center(1:dim_num);
    dist = sqrt ( sum ( ( p(1:dim_num) - pn(1:dim_num) ).^2 ) );
    return
  end
%
%  If the test point is at the center, then the computation is easy.
%  The angle subtended by any side is ( 2 * PI / NSIDE ) and the
%  nearest distance is the midpoint of any such side.
%
  if ( p(1:dim_num) == center(1:dim_num) )
    angle = pi / nside;
    pd(1) =   ( p(1) - center(1) ) * cos ( angle ) ...
            + ( p(2) - center(2) ) * sin ( angle );
    pd(2) = - ( p(1) - center(1) ) * sin ( angle ) ...
            + ( p(2) - center(2) ) * cos ( angle );
    pn(1) = center(1) + pd(1) * cos ( angle );
    pn(2) = center(2) + pd(2) * sin ( angle );
    dist = radius * cos ( angle );
    return
  end
%
%  Determine the angle between the ray to the first corner,
%  and the ray to the test point.
%
  angle = angle_deg_2d ( p1(1:2), center(1:2), p(1:2) );
%
%  Determine the sector of the point.
%
  sector_index = floor ( angle / sector_angle ) + 1;
%
%  Generate the two corner points that terminate the SECTOR-th side.
%
  angle2 = ( sector_index - 1 ) * sector_angle;
  angle2 = degrees_to_radians ( angle2 );

  pa = vector_rotate_base_2d ( p1, center, angle2 );

  angle2 = ( sector_index ) * sector_angle;
  angle2 = degrees_to_radians ( angle2 );

  pb = vector_rotate_base_2d ( p1, center, angle2 );
%
%  Determine the point on the SECTOR-th side of the shape which is
%  nearest.
%
  [ pn, dist, t ] = segment_point_near_2d ( pa, pb, p );

  return
end
