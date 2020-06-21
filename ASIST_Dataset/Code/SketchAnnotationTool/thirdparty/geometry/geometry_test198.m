function geometry_test198 ( )

%*****************************************************************************80
%
%% TEST198 tests SHAPE_POINT_NEAR_2D.
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
  center(1:dim_num) = [ 3.0, 0.0 ];
  p1(1:dim_num) = [ 5.0, 0.0 ];
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
  fprintf ( 1, 'TEST198\n' );
  fprintf ( 1, '  For a shape in 2D,\n' );
  fprintf ( 1, '  SHAPE_POINT_NEAR_2D computes the nearest\n' );
  fprintf ( 1, '    point to a point;\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  Number of sides:\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  %d\n', nside );

  r8vec_print ( dim_num, center, '  Hexagon center:' );

  fprintf ( 1, '\n' );
  fprintf ( 1, '  Hexagon vertex #1:\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  %8f  %8f\n', p1(1:dim_num) );
  fprintf ( 1, '\n' );
  fprintf ( 1, '     I       X            Y              PN     Dist\n' );
  fprintf ( 1, '\n' );

  for i = 1 : ntest

    p(1:dim_num) = ptest(1:dim_num,i);

    [ pn, dist ] = shape_point_near_2d ( center, p1, nside, p ) ;

    fprintf ( 1, '  %6d  %10f  %10f  %10f  %10f  %10f\n', ...
      i, p(1:dim_num), pn(1:dim_num), dist );

  end
 
  return
end
