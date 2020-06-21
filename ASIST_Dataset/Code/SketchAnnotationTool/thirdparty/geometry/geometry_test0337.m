function geometry_test0337 ( )

%*****************************************************************************80
%
%% TEST0337 tests LINE_EXP_POINT_DIST_SIGNED_2D.
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
  ntest = 3;

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST0337\n' );
  fprintf ( 1, '  LINE_EXP_POINT_DIST_SIGNED_2D finds the signed\n' );
  fprintf ( 1, '    distance to a point from an explicit line.\n' );

  p1(1:dim_num) = [ 1.0, 3.0 ];
  p2(1:dim_num) = [ 4.0, 0.0 ];

  ptest(1:dim_num,1:ntest) = [ ...
    0.0,  0.0; ...
    5.0, -1.0; ...
    5.0,  3.0 ]';

  r8vec_print ( dim_num, p1, '  Point 1: ' );
  r8vec_print ( dim_num, p2, '  Point 2: ' );

  for i = 1 : ntest

    p(1:dim_num) = ptest(1:dim_num,i);

    r8vec_print ( dim_num, p, '  Point: ' );

    dist = line_exp_point_dist_signed_2d ( p1, p2, p );

    fprintf ( 1, '\n' );
    fprintf ( 1, '  Signed distance = %f\n', dist );

  end

  return
end
