function normal = polygon_normal_3d ( n, v )

%*****************************************************************************80
%
%% POLYGON_NORMAL_3D computes the normal vector to a polygon in 3D.
%
%  Discussion:
%
%    If the polygon is planar, then this calculation is correct.
%
%    Otherwise, the normal vector calculated is the simple average
%    of the normals defined by the planes of successive triples
%    of vertices.
%
%    If the polygon is "almost" planar, this is still acceptable.
%    But as the polygon is less and less planar, so this averaged normal
%    vector becomes more and more meaningless.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    12 August 2005
%
%  Author:
%
%    John Burkardt
%
%  Reference:
%
%    Paulo Cezar Pinto Carvalho and Paulo Roma Cavalcanti,
%    Point in Polyhedron Testing Using Spherical Polygons,
%    in Graphics Gems V,
%    edited by Alan Paeth,
%    Academic Press, 1995, T385.G6975.
%
%  Parameters:
%
%    Input, integer N, the number of vertices.
%
%    Input, real V(3,N), the coordinates of the vertices.
%
%    Output, real NORMAL(3), the averaged normal vector
%    to the polygon.
%
  dim_num = 3;

  normal(1:dim_num) = 0.0;

  v1(1:dim_num) = v(1:dim_num,2) - v(1:dim_num,1);

  for j = 3 : n

    v2(1:dim_num) = v(1:dim_num,j) - v(1:dim_num,1);

    p = r8vec_cross_3d ( v1, v2 );

    normal(1:dim_num) = normal(1:dim_num) + p(1:dim_num);

    v1(1:dim_num) = v2(1:dim_num);

  end
%
%  Normalize.
%
  normal_norm = r8vec_length ( dim_num, normal );

  if ( normal_norm == 0.0 )
    return
  end

  normal(1:dim_num) = normal(1:dim_num) / normal_norm;

  return
end
