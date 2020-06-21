function geometry_test063 ( )

%*****************************************************************************80
%
%% TEST063 tests PLANE_NORMAL2EXP_3D.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    07 April 2009
%
%  Author:
%
%    John Burkardt
%
  dim_num = 3;

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST063\n' );
  fprintf ( 1, '  PLANE_NORMAL2EXP_3D puts a plane defined by\n' );
  fprintf ( 1, '    point, normal form into explicit form.\n' );

  pp(1:dim_num) = [ -1.0,   0.0, -1.0 ];
  normal(1:dim_num) = [ -0.2672612, -0.5345225, -0.8017837 ];

  r8vec_print ( dim_num, pp, '  The point PP:' );

  r8vec_print ( dim_num, normal, '  Normal vector:' );
 
  [ p1, p2, p3 ] = plane_normal2exp_3d ( pp, normal );
 
  fprintf ( 1, '\n' );
  fprintf ( 1, '  P1, P2, P3:\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  %12f  %12f  %12f\n', p1(1:dim_num) );
  fprintf ( 1, '  %12f  %12f  %12f\n', p2(1:dim_num) );
  fprintf ( 1, '  %12f  %12f  %12f\n', p3(1:dim_num) );

  return
end
