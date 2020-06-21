function center = triangle_circumcenter_2d ( t )

%*****************************************************************************80
%
%% TRIANGLE_CIRCUMCENTER_2D computes the circumcenter of a triangle in 2D.
%
%  Discussion:
%
%    The circumcenter of a triangle is the center of the circumcircle, the
%    circle that passes through the three vertices of the triangle.
%
%    The circumcircle contains the triangle, but it is not necessarily the
%    smallest triangle to do so.
%
%    If all angles of the triangle are no greater than 90 degrees, then
%    the center of the circumscribed circle will lie inside the triangle.
%    Otherwise, the center will lie outside the triangle.
%
%    The circumcenter is the intersection of the perpendicular bisectors
%    of the sides of the triangle.
%
%    In geometry, the circumcenter of a triangle is often symbolized by "O".
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    09 February 2005
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real T(2,3), the triangle vertices.
%
%    Output, real CENTER(2), the circumcenter of the triangle.
%
  dim_num = 2;

  f(1) = ( t(1,2) - t(1,1) ).^2 + ( t(2,2) - t(2,1) ).^2;
  f(2) = ( t(1,3) - t(1,1) ).^2 + ( t(2,3) - t(2,1) ).^2;
  
  top(1) =    ( t(2,3) - t(2,1) ) * f(1) - ( t(2,2) - t(2,1) ) * f(2);
  top(2) =  - ( t(1,3) - t(1,1) ) * f(1) + ( t(1,2) - t(1,1) ) * f(2);

  det  =    ( t(2,3) - t(2,1) ) * ( t(1,2) - t(1,1) ) ...
          - ( t(2,2) - t(2,1) ) * ( t(1,3) - t(1,1) ) ;

  center(1:2) = t(1:2,1)' + 0.5 * top(1:2) / det;

  return
end
