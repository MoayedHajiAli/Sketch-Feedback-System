function geometry_test02035 ( )

%*****************************************************************************80
%
%% TEST02035 tests CYLINDER_SAMPLE_3D.
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
  n = 20;

  p1 = [ 0.0, -2.0, 0.0 ];
  p2 = [ 0.0,  2.0, 0.0 ];
  r = 1.0;
  seed = 123456789;

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST02035\n' );
  fprintf ( 1, '  CYLINDER_SAMPLE_3D samples points in a cylinder.\n' );

  fprintf ( 1, '\n' );
  fprintf ( 1, '  Radius R = %f\n', r );
  fprintf ( 1, '  Center of bottom disk = %f  %f  %f\n', p1(1:dim_num) );
  fprintf ( 1, '  Center of top disk =    %f  %f  %f\n', p2(1:dim_num) );

  [ p, seed ] = cylinder_sample_3d ( p1, p2, r, n, seed );

  r8mat_transpose_print ( dim_num, n, p, '  Sample points:' );

  return
end
