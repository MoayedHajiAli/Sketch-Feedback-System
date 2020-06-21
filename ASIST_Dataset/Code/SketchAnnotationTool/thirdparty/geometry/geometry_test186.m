function geometry_test186 ( )

%*****************************************************************************80
%
%% TEST186 tests SPHERE_IMP_GRIDLINES_3D.
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
  maxline = 1000;

  nlat = 3;
  nlong = 4;

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST186\n' );
  fprintf ( 1, '  SPHERE_IMP_GRIDLINES_3D computes gridlines\n' );
  fprintf ( 1, '  on a sphere in 3D.\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  Number of intermediate latitudes is %d\n', nlat );
  fprintf ( 1, '  Number of longitudes is %d\n', nlong );

  [ nline, line ] = sphere_imp_gridlines_3d ( maxline, nlat, nlong );

  fprintf ( 1, '\n' );
  fprintf ( 1, 'Number of line segments is %d\n', nline );

  i4mat_transpose_print ( 2, nline, line, '  Grid line vertices:' );

  return
end
