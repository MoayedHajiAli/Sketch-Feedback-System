function geometry_test0165 ( )

%*****************************************************************************80
%
%% TEST0165 tests CIRCLE_IMP_POINTS_3D.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    14 February 2003
%
%  Author:
%
%    John Burkardt
%
  dim_num = 3;
  n = 12;

  nc = [ 1.0, 1.0, 1.0 ];
  pc = [ 5.0, -2.0, 1.0 ];
  r = 2.0;

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST0165\n' );
  fprintf ( 1, '  CIRCLE_IMP_POINTS_3D gets points on a circle in 3D;\n' );

  circle_imp_print_3d ( r, pc, nc, '  The implicit circle:' );

  p = circle_imp_points_3d ( r, pc, nc, n );

  r8mat_transpose_print ( dim_num, n, p, '  Points on the circle:' );

  return
end
