function geometry_test2036 ( )

%*****************************************************************************80
%
%% TEST2036 tests THETA3_ADJUST.
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
  fprintf ( 1, 'TEST2036\n' );
  fprintf ( 1, '  THETA3_ADJUST tries to "adjust" the THETA coordinates\n' );
  fprintf ( 1, '  of three points by adding 2 PI where necessary, to\n' );
  fprintf ( 1, '  reduce the range, that is, MAX(T1,T2,T3) - MIN(T1,T2,T3).\n' );

  r8_lo = 0.0;
  r8_hi = 2.0 * pi;

  fprintf ( 1, '\n' );
  fprintf ( 1, '     Index              Theta1      Theta2      Theta3      Range\n' );
  fprintf ( 1, '\n' );

  for test = 1 : test_num

    [ theta1, seed ] = r8_uniform ( r8_lo, r8_hi, seed );
    [ theta2, seed ] = r8_uniform ( r8_lo, r8_hi, seed );
    [ theta3, seed ] = r8_uniform ( r8_lo, r8_hi, seed );

    theta_range = max ( theta1, max ( theta2, theta3 ) ) ...
                - min ( theta1, min ( theta2, theta3 ) );

    fprintf ( 1, '\n' );
    fprintf ( 1, '  %8d  Before  %10f  %10f  %10f  %10f\n', ...
      test, theta1, theta2, theta3, theta_range );

    [ theta1, theta2, theta3 ] = theta3_adjust ( theta1, theta2, theta3 );

    theta_range = max ( theta1, max ( theta2, theta3 ) ) ...
                - min ( theta1, min ( theta2, theta3 ) );

    fprintf ( 1, '  %8d  After   %10f  %10f  %10f  %10f\n', ...
      test, theta1, theta2, theta3, theta_range );

  end

  return
end
