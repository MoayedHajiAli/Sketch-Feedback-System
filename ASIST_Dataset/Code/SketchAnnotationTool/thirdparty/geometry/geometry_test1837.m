function geometry_test1837 ( )

%*****************************************************************************80
%
%% TEST1837 tests SPHERE_IMP_GRID_ICOS_SIZE.
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
  fprintf ( 1, 'TEST1837\n' );
  fprintf ( 1, '  SPHERE_IMP_GRID_ICOS_SIZE determines the size\n' );
  fprintf ( 1, '  (number of nodes, number of triangles) in a grid\n' );
  fprintf ( 1, '  on a sphere, made by subdividing an initial\n' );
  fprintf ( 1, '  projected icosahedron.\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  FACTOR determines the number of subdivisions of each\n' );
  fprintf ( 1, '  edge of the icosahedral faces.\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '    FACTOR     Nodes     Edges Triangles\n' );
  fprintf ( 1, '  --------  --------  --------  --------\n' );
  fprintf ( 1, '\n' );

  for factor = 1 : 10
    [ node_num, edge_num, triangle_num ] = sphere_imp_grid_icos_size ( factor );
    fprintf ( 1, '  %8d  %8d  %8d  %8d\n', ...
      factor, node_num, edge_num, triangle_num );
  end

  return
end
