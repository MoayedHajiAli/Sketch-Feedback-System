function geometry_test199 ( )

%*****************************************************************************80
%
%% TEST199 tests SHAPE_RAY_INT_2D.
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
  ntest = 4;

  center = [ 3.0, 0.0 ];
  p1 = [ 5.0, 0.0 ];
  pa_test = [ ...
    3.0,  0.0; ...
    3.0,  0.0; ...
    3.0, -1.0; ...
    3.0, -1.0 ]';
  pb_test = [ ...
    4.0,  0.0; ...
    3.0,  1.0; ...
    3.0,  1.0; ...
    7.0,  5.0 ]';

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST199\n' );
  fprintf ( 1, '  For a shape in 2D,\n' );
  fprintf ( 1, '  SHAPE_RAY_INT_2D computes the intersection of\n' );
  fprintf ( 1, '    a shape and a ray whose origin is within\n' );
  fprintf ( 1, '    the shape.\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  Number of sides:\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  %d\n', nside );

  r8vec_print ( dim_num, center, '  Hexagon center:' );

  fprintf ( 1, '\n' );
  fprintf ( 1, '  Hexagon vertex #1:\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  %12f  %12f\n', p1(1:dim_num) );
  fprintf ( 1, '\n' );
  fprintf ( 1, '     I       XA          YA          ' );
  fprintf ( 1, 'XB          YB          XI          YI\n' );
  fprintf ( 1, '\n' );

  for i = 1 : ntest

    pa(1:dim_num) = pa_test(1:dim_num,i);
    pb(1:dim_num) = pb_test(1:dim_num,i);

    pint = shape_ray_int_2d ( center, p1, nside, pa, pb );

    fprintf ( 1, '  %6d  %10f  %10f  %10f  %10f  %10f  %10f\n', ...
      i, pa(1:dim_num), pb(1:dim_num), pint(1:dim_num) );

  end
 
  return
end
