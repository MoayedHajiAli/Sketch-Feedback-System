function geometry_test223 ( )

%*****************************************************************************80
%
%% TEST223 tests VECTOR_ROTATE_BASE_2D;
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    18 February 2009
%
%  Author:
%
%    John Burkardt
%
  dim_num = 2;
  ntest = 4;

  atest = [ 30.0, -45.0, 270.0, 20.0 ];
  pb = [ 10.0, 5.0 ];
  ptest = [
    11.0, 5.0; ...
    10.0, 7.0; ...
    11.0, 6.0; ...
    10.0, 5.0 ]';

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST223\n' );
  fprintf ( 1, '  VECTOR_ROTATE_BASE_2D rotates a vector (X1,Y1)\n' );
  fprintf ( 1, '  through an angle around a base point (XB,YB).\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '        P1              PB       Angle          P2\n' );
  fprintf ( 1, '\n' );

  for i = 1 : ntest

    p1 = ptest(1:dim_num,i);

    angle = degrees_to_radians ( atest(i) );

    p2 = vector_rotate_base_2d ( p1, pb, angle );

    fprintf ( 1, '  %10f  %10f  %10f  %10f  %10f  %10f  %10f\n', ...
      p1(1:dim_num), pb(1:dim_num), atest(i), p2(1:dim_num) );
 
  end
 
  return
end
