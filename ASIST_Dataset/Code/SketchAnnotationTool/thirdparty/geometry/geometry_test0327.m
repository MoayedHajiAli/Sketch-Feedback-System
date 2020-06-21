function geometry_test0327 ( )

%*****************************************************************************80
%
%% TEST0327 tests LINE_EXP_NORMAL_2D.
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
  dim_num = 2;
  ntest = 3;

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST0327\n' );
  fprintf ( 1, '  LINE_EXP_NORMAL_2D determines a unit normal vector\n' );
  fprintf ( 1, '  to a given explicit line.\n' );

  p1(1:dim_num) = [ 1.0, 3.0 ];
  p2(1:dim_num) = [ 4.0, 0.0 ];

  r8vec_print ( dim_num, p1, '  Point 1: ' );
  r8vec_print ( dim_num, p2, '  Point 2: ' );

  normal = line_exp_normal_2d ( p1, p2 );

  r8vec_print ( dim_num, normal, '  Normal vector N:' );

  return
end
