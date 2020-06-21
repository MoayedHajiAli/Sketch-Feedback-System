function geometry_test005 ( )

%*****************************************************************************80
%
%% TEST005 tests ATAN4;
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
  test_num = 8;

  x_test = [ 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 0.0 ];
  y_test = [ 0.0, 1.0, 2.0, 0.0, -1.0, -1.0, -1.0, -1.0 ];

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST005\n' );
  fprintf ( 1, '  ATAN4 computes an angle with a given tangent.\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  X, Y, ATAN(Y/X), ATAN2(Y,X), ATAN4(Y,X)\n' );
  fprintf ( 1, '\n' );

  for i = 1 : test_num

    x = x_test(i);
    y = y_test(i);

    if ( x ~= 0.0 )
      temp1 = atan ( y / x );
    else
      temp1 = Inf;
    end

    temp2 = atan2 ( y, x );
    temp3 = atan4 ( y, x );
    
    fprintf ( 1, '  %12f  %12f  %12f  %12f  %12f\n', ...
      x, y, temp1, temp2, temp3 );
 
  end
 
  fprintf ( 1, '\n' );
  fprintf ( 1, '  Repeat, but display answers in degrees.\n' );
  fprintf ( 1, '\n' );

  for i = 1 : test_num

    x = x_test(i);
    y = y_test(i);

    if ( x ~= 0.0 )
      temp1 = radians_to_degrees ( atan ( y / x ) );
    else
      temp1 = Inf;
    end
    
    temp2 = radians_to_degrees ( atan2 ( y, x ) );
    temp3 = radians_to_degrees ( atan4 ( y, x ) );
    
    fprintf ( 1, '  %12f  %12f  %12f  %12f  %12f\n', ...
      x, y, temp1, temp2, temp3 );
 
  end
 
  return
end
