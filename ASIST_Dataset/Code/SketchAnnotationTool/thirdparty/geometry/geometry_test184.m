function geometry_test184 ( )

%*****************************************************************************80
%
%% TEST184 tests SPHERE_IMP_GRIDPOINTS_3D.
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
  dim_num = 3;
  maxpoint = 100;

  r = 10.0;

  center(1:dim_num) = [ 0.0, 0.0, 0.0 ];

  nlat = 3;
  nlong = 4;

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST184\n' );
  fprintf ( 1, '  SPHERE_IMP_GRIDPOINTS_3D produces a grid of\n' );
  fprintf ( 1, '  points on an implicit sphere in 3D.\n' );

  fprintf ( 1, '\n' );
  fprintf ( 1, '  Radius %f\n', r );

  r8vec_print ( dim_num, center, '  Center:' )

  fprintf ( 1, '\n' );
  fprintf ( 1, '  The number of intermediate latitudes = %d\n', nlat );
  fprintf ( 1, '  The number of longitudes = %d\n', nlong );

  [ npoint, p ] = sphere_imp_gridpoints_3d ( r, center, maxpoint, nlat, nlong );

  fprintf ( 1, '\n' );
  fprintf ( 1, '  The number of grid points is %d\n', npoint );
  fprintf ( 1, '\n' );

  k = 1;
  fprintf ( 1, '  %6d  %12f  %12f  %12f\n', k, p(1:dim_num,k) );

  for i = 1 : nlat
    fprintf ( 1, '\n' );
    for j = 0 : nlong - 1
      k = k + 1;
      fprintf ( 1, '  %6d  %12f  %12f  %12f\n', k, p(1:dim_num,k) );
    end
  end

  k = k + 1;
  fprintf ( 1, '\n' );
  fprintf ( 1, '  %6d  %12f  %12f  %12f\n', k, p(1:dim_num,k) );

  return
end
