function geometry_test015 ( )

%*****************************************************************************80
%
%% TEST015 tests CIRCLE_EXP2IMP_2D, TRIANGLE_CIRCUMCIRCLE_2D.
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
  n = 3;

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST015\n' );
  fprintf ( 1, '  CIRCLE_EXP2IMP_2D computes the radius and \n' );
  fprintf ( 1, '    center of the circle through three points.\n' );
  fprintf ( 1, '  TRIANGLE_CIRCUMCIRCLE_2D computes the radius and \n' );
  fprintf ( 1, '    center of the circle through the vertices of\n' );
  fprintf ( 1, '    a triangle.\n' );

  p1test(1:dim_num,1:n) = [ ...
    4.0, 2.0; ...
    1.0, 5.0; ...
   -2.0, 2.0 ]';

  p2test(1:dim_num,1:n) = [ ...
    4.0, 2.0; ...
    5.0, 4.0; ...
    6.0, 6.0 ]';

  p3test(1:dim_num,1:n) = [ ...
    4.0, 2.0; ...
    1.0, 5.0; ...
    4.0, 2.0 ]';

  for i = 1 : n

    p1(1:dim_num) = p1test(1:dim_num,i);
    p2(1:dim_num) = p2test(1:dim_num,i);
    p3(1:dim_num) = p3test(1:dim_num,i);

    r8vec_print ( dim_num, p1, '  P1:' );
    r8vec_print ( dim_num, p2, '  P2:' );
    r8vec_print ( dim_num, p3, '  P3:' );

    [ r, center ] = circle_exp2imp_2d ( p1, p2, p3 );

    circle_imp_print_2d ( r, center, '  The implicit circle:' )

    t(1:dim_num,1:3) = [ p1(1:dim_num); p2(1:dim_num); p3(1:dim_num) ]';

    [ r, center ] = triangle_circumcircle_2d ( t );

    circle_imp_print_2d ( r, center, '  The triangle''s circumcircle:' )
 
  end

  return
end
