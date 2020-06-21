function geometry_test1838 ( )

%*****************************************************************************80
%
%% TEST1838 tests SPHERE_IMP_GRIDPOINTS_ICOS1.
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
  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST1838\n' );
  fprintf ( 1, '  SPHERE_IMP_GRID_ICOS_SIZE "sizes" a grid generated\n' );
  fprintf ( 1, '  on an icosahedron and projected to a sphere.\n' );
  fprintf ( 1, '  SPHERE_IMP_GRIDPOINTS_ICOS1 creates the grid points.\n' );

  factor = 2;

  fprintf ( 1, '\n' );
  fprintf ( 1, '  Sizing factor FACTOR = %d\n', factor );

  [ node_num, edge_num, triangle_num ] = sphere_imp_grid_icos_size ( factor );

  fprintf ( 1, '\n' );
  fprintf ( 1, '  Number of nodes =     %d\n', node_num );
  fprintf ( 1, '  Number of edges =     %d\n', edge_num );
  fprintf ( 1, '  Number of triangles = %d\n', triangle_num );

  node_xyz = sphere_imp_gridpoints_icos1 ( factor, node_num );

  r8mat_transpose_print_some ( 3, node_num, node_xyz, 1, 1, 3, 10, ...
    '  Initial part of NODE_XYZ array:' );

  return
end
