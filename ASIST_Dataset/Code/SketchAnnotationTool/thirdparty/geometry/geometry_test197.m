function geometry_test197 ( )

%*****************************************************************************80
%
%% TEST197 tests SHAPE_POINT_DIST_2D.
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
  dim_num = 2;
  nside = 6;
  ntest = 8;

  center = [ 3.0, 0.0 ];
  p1 = [ 5.0, 0.0 ];
  ptest(1:dim_num,1:ntest) = [ ...
     3.0, 0.0; ...
     5.0, 0.0; ...
     4.0, 0.0; ...
    10.0, 0.0; ...
     4.0, 1.7320508; ...
     5.0, 2.0 * 1.7320508; ...
     3.0, 1.7320508; ...
     3.0, 1.7320508 / 2.0 ]';

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST197\n' );
  fprintf ( 1, '  For a shape in 2D,\n' );
  fprintf ( 1, '  SHAPE_POINT_DIST_2D computes the distance\n' );
  fprintf ( 1, '    to a point;\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  Number of sides:\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  %d\n', nside );

  r8vec_print ( dim_num, center, '  Center of hexagon:' );

  r8vec_print ( dim_num, p1, '  Hexagon vertex #1' );

  fprintf ( 1, '\n' );
  fprintf ( 1, '     I	   X		Y	     DIST\n' );
  fprintf ( 1, '\n' );

  for i = 1 : ntest

    p(1:dim_num) = ptest(1:dim_num,i);

    dist = shape_point_dist_2d ( center, p1, nside, p ) ;

    fprintf ( 1, '  %6d  %12f  %12f  %12f\n', i, p(1:dim_num), dist );

  end
 
  return
end
