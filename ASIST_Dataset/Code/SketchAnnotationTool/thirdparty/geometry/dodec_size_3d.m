function [ point_num, edge_num, face_num, face_order_max ] = ...
  dodec_size_3d ( )

%*****************************************************************************80
%
%% DODEC_SIZE_3D gives "sizes" for a dodecahedron in 3D.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    22 July 2007
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Output, integer POINT_NUM, the number of points.
%
%    Output, integer EDGE_NUM, the number of edges.
%
%    Output, integer FACE_NUM, the number of faces.
%
%    Output, integer FACE_ORDER_MAX, the maximum order of any face.
%
  point_num = 20;
  edge_num = 30;
  face_num = 12;
  face_order_max = 5;

  return
end
