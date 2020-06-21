function node_xyz = sphere_imp_gridpoints_icos2 ( factor, node_num )

%*****************************************************************************80
%
%% SPHERE_IMP_GRIDPOINTS_ICOS2 returns icosahedral grid points on a sphere.
%
%  Discussion:
%
%    With FACTOR = 1, the grid has 20 triangular faces and 12 nodes.
%
%    With FACTOR = 2, each triangle of the icosahedron is subdivided into
%    2x2 subtriangles, resulting in 80 faces and 
%    42 = 12 + 20 * 3 * (1)/2 + 20 * 0 ) nodes.
%
%    With FACTOR = 3, each triangle of the icosahedron is subdivided into
%    3x3 subtriangles, resulting in 180 faces and 
%    72 ( = 12 + 20 * 3 * (2)/2 + 20 * 1 ) nodes.
%
%    In general, each triangle is subdivided into FACTOR*FACTOR subtriangles,
%    resulting in 20 * FACTOR * FACTOR faces and
%      12 
%    + 20 * 3          * (FACTOR-1) / 2 
%    + 20 * (FACTOR-2) * (FACTOR-1) / 2 nodes.
%
%
%    There are two possible ways of doing the subdivision:
%
%    If we subdivide the secants, we will be creating congruent faces and
%    sides on the original, non-projected icosahedron, which will result,
%    after projection, in faces and sides on the sphere that are not congruent.
%
%    If we subdivide the spherical angles, then after projection we will 
%    have spherical faces and sides that are congruent.  In general, this
%    is likely to be the more desirable subdivision scheme.
%
%    This routine uses the angle subdivision scheme.
%
%    NOTE: Despite my initial optimism, THETA2_ADJUST and THETA3_ADJUST 
%    do not seem to have enough information to properly adjust the values of
%    THETA when the edge or triangle includes the north or south poles as a 
%    vertex or in the interior.  Of course, if a pole is a vertex, its THETA
%    value is meaningless, and this routine will be deceived by trying
%    to handle a meaningless THETA=0 value.  I will need to think some
%    more to properly handle the spherical coordinates when I want to
%    interpolate.  Until then, the results of this routine are INCORRECT.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    23 July 2007
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer FACTOR, the subdivision factor, which must
%    be at least 1.
%
%    Input, integer NODE_NUM, the number of nodes, as reported
%    by SPHERE_IMP_GRID_ICOS_SIZE.
%
%    Output, real NODE_XYZ(3,NODE_NUM), the node coordinates.
%
%  Local Parameters:
%
%    POINT_NUM, EDGE_NUM, FACE_NUM and FACE_ORDER_MAX are counters 
%    associated with the icosahedron, and POINT_COORD, EDGE_POINT, 
%    FACE_ORDER and FACE_POINT are data associated with the icosahedron.
%    We need to refer to this data to generate the grid.
%
%    NODE counts the number of nodes we have generated so far.  At the
%    end of the routine, it should be equal to NODE_NUM.
%
  r8_1 = 1.0;
%
%  Size the icosahedron.
%
  [ point_num, edge_num, face_num, face_order_max ] = icos_size_3d ( );
%
%  Set the icosahedron.
%
  [ point_coord, edge_point, face_order, face_point ] = icos_shape_3d ( ...
    point_num, edge_num, face_num, face_order_max );
%
%  Generate the point coordinates.
%
%  A.  Points that are the icosahedral vertices.
%
  node = 0;
  node_xyz(1:3,1:point_num) = point_coord(1:3,1:point_num);
%
%  B. Points in the icosahedral edges, at 
%  1/FACTOR, 2/FACTOR, ..., (FACTOR-1)/FACTOR.
%
  node = 12;

  for edge = 1 : edge_num

    a = edge_point(1,edge);
    [ a_r, a_t, a_p ] = xyz_to_rtp ( point_coord(1:3,a) );

    b = edge_point(2,edge);
    [ b_r, b_t, b_p ] = xyz_to_rtp ( point_coord(1:3,b) );

    [ a_t, b_t ] = theta2_adjust ( a_t, b_t );

    for f = 1 : factor - 1

      node = node + 1;

      t = ...
        ( ( factor - f ) * a_t   ...
        + (          f ) * b_t ) ...
        / ( factor     );

      p = ...
        ( ( factor - f ) * a_p   ...
        + (          f ) * b_p ) ...
        / ( factor     );

      node_xyz(1:3,node) = rtp_to_xyz ( r8_1, t, p );

    end
  end
%
%  C.  Points in the icosahedral faces.
%
  for face = 1 : face_num

    a = face_point(1,face);
    [ a_r, a_t, a_p ] = xyz_to_rtp ( point_coord(1:3,a) );

    b = face_point(2,face);
    [ b_r, b_t, b_p ] = xyz_to_rtp ( point_coord(1:3,b) );

    c = face_point(3,face);
    [ c_r, c_t, c_p ] = xyz_to_rtp ( point_coord(1:3,c) );

    [ a_t, b_t, c_t ] = theta3_adjust ( a_t, b_t, c_t );

    for f1 = 1 : factor - 2
      for f2 = 1 : factor - f1 - 1

        node = node + 1;

        t = ...
          ( ( factor - f1 - f2 ) * a_t   ...
          + (          f1      ) * b_t   ...
          + (               f2 ) * c_t ) ...
          / ( factor           );

        p = ...
          ( ( factor - f1 - f2 ) * a_p   ...
          + (          f1      ) * b_p   ...
          + (               f2 ) * c_p ) ...
          / ( factor           );

        node_xyz(1:3,node) = rtp_to_xyz ( r8_1, t, p );

      end
    end

  end

  return
end
