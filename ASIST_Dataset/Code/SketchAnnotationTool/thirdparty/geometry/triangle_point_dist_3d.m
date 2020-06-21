function dist = triangle_point_dist_3d ( t, p )

%*****************************************************************************80
%
%% TRIANGLE_POINT_DIST_3D: distance ( triangle, point ) in 3D.
%
%  Discussion:
%
%    Thanks to Ozgur Ozturk for pointing out that the triangle vertices
%    needed to be transposed before being passed to SEGMENT_POINT_DIST_3D,
%    04 May 2005.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    04 May 2005
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real T(3,3), the triangle vertices.
%
%    Input, real P(3), the point which is to be checked.
%
%    Output, real DIST, the distance from the point to the
%    triangle.  DIST is zero if the point lies exactly on the triangle.
%
  dim_num = 3;
%
%  Compute the distances from the point to each of the sides.
%
  dist2 = segment_point_dist_3d ( t(1:dim_num,1)', t(1:dim_num,2)', p );

  dist = dist2;

  dist2 = segment_point_dist_3d ( t(1:dim_num,2)', t(1:dim_num,3)', p );

  dist = min ( dist, dist2 );

  dist2 = segment_point_dist_3d ( t(1:dim_num,3)', t(1:dim_num,1)', p );

  dist = min ( dist, dist2 );

  return
end
