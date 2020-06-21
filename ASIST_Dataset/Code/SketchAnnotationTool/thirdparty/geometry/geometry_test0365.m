function geometry_test0365 ( )

%*****************************************************************************80
%
%% TEST0365 tests SEGMENT_POINT_DIST_2D.
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
  test_num = 3;
  seed = 123456789;

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST0365\n' );
  fprintf ( 1, '  SEGMENT_POINT_DIST_2D computes the distance\n' );
  fprintf ( 1, '    between a line segment and point in 2D.\n' );

  for test = 1 : test_num

    [ p1, seed ] = r8vec_uniform_01 ( dim_num, seed );
    [ p2, seed ] = r8vec_uniform_01 ( dim_num, seed );
    [ p, seed ] = r8vec_uniform_01 ( dim_num, seed );

    dist = segment_point_dist_2d ( p1, p2, p );

    fprintf ( 1, '\n' );
    fprintf ( 1, '  TEST = %d', test );
    fprintf ( 1, '  P1 =   %12f  %12f\n', p1(1:dim_num) );
    fprintf ( 1, '  P2 =   %12f  %12f\n', p2(1:dim_num) );
    fprintf ( 1, '  P =    %12f  %12f\n', p(1:dim_num) );
    fprintf ( 1, '  DIST = %12f\n', dist );

  end

  return
end
