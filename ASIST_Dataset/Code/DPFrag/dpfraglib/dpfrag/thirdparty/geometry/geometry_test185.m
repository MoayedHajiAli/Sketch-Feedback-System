function geometry_test185 ( )

%*****************************************************************************80
%
%% TEST185 tests SPHERE_IMP_SPIRALPOINTS_3D.
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
  n = 20;
  dim_num = 3;

  r = 1.0;
  center(1:dim_num) = [ 0.0, 0.0, 0.0 ];

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST185\n' );
  fprintf ( 1, '  SPHERE_IMP_SPIRALPOINTS_3D produces a spiral of\n' );
  fprintf ( 1, '  points on an implicit sphere in 3D.\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  Radius %f\n', r );

  r8vec_print ( dim_num, center, '  Center:' );

  fprintf ( 1, '\n' );
  fprintf ( 1, '  The number of spiral points is %d\n', n );

  p = sphere_imp_spiralpoints_3d ( r, center, n );

  r8mat_transpose_print ( dim_num, n, p, '  The spiral points:' );

  return
end
