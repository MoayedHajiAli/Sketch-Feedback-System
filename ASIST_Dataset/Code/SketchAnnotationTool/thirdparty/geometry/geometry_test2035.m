function geometry_test2035 ( )

%*****************************************************************************80
%
%% TEST2035 tests THETA2_ADJUST.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    22 July 2007
%
%  Author:
%
%    John Burkardt
%
  seed = 123456789;
  test_num = 10;

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST2035\n' );
  fprintf ( 1, '  THETA2_ADJUST tries to "adjust" the THETA coordinates\n' );
  fprintf ( 1, '  of two points by adding 2 PI where necessary, to\n' );
  fprintf ( 1, '  reduce the range, that is, MAX(T1,T2) - MIN(T1,T2).\n' );

  r8_lo = 0.0;
  r8_hi = 2.0 * pi;

  fprintf ( 1, '\n' );
  fprintf ( 1, '     Index              Theta1      Theta2      Range\n' );
  fprintf ( 1, '\n' );

  for test = 1 : test_num

    [ theta1, seed ] = r8_uniform ( r8_lo, r8_hi, seed );
    [ theta2, seed ] = r8_uniform ( r8_lo, r8_hi, seed );

    theta_range = max ( theta1, theta2 ) - min ( theta1, theta2 );

    fprintf ( 1, '\n' );
    fprintf ( 1, '  %8d  Before  %10f  %10f  %10f\n', test, theta1, theta2, theta_range );

    [ theta1, theta2 ] = theta2_adjust ( theta1, theta2 );

    theta_range = max ( theta1, theta2 ) - min ( theta1, theta2 );

    fprintf ( 1, '  %8d  After   %10f  %10f  %10f\n', test, theta1, theta2, theta_range );

  end

  return
end
