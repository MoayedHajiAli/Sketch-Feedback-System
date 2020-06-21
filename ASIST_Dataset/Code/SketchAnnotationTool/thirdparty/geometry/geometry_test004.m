function geometry_test004 ( )

%*****************************************************************************80
%
%% TEST004 tests ARC_COSINE;
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
  test_num = 9;

  test_x = [ 5.0, 1.2, 1.0, 0.9, 0.5, 0.0, -0.9, -1.0, -1.01 ];

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST004\n' );
  fprintf ( 1, '  ARC_COSINE computes an angle with a given cosine;\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  X, ARC_COSINE(X), (Degrees)\n' );
  fprintf ( 1, '\n' );

  for i = 1 : test_num

    x = test_x(i);

    temp1 = arc_cosine ( x );
    temp2 = radians_to_degrees ( temp1 );

    fprintf ( 1, '  %8f  %12f  %12f\n', x, temp1, temp2 );
 
  end
 
  return
end
