function geometry_test2101 ( )

%*****************************************************************************80
%
%% TEST2101 tests TRIANGLE_CIRCUMCENTER_2D, TRIANGLE_CIRCUMCENTER_2D_2;
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    19 February 2009
%
%  Author:
%
%    John Burkardt
%
  dim_num = 2;
  ntest = 4;

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST2101\n' );
  fprintf ( 1, '  For a triangle in 2D:\n' );
  fprintf ( 1, '  TRIANGLE_CIRCUMCENTER_2D computes the circumcenter.\n' );
  fprintf ( 1, '  TRIANGLE_CIRCUMCENTER_2D_2 computes the circumcenter.\n' );

  for i = 1 : ntest

    if ( i == 1 )
      t(1:dim_num,1:3) = [ ...
         10.0,  5.0; ...
         11.0,  5.0; ...
         10.0,  6.0 ]';
    elseif ( i == 2 )
      t(1:dim_num,1:3) = [ ...
         10.0,  5.0; ...
         11.0,  5.0; ...
         10.5,  5.86602539 ]';
    elseif ( i == 3 )
      t(1:dim_num,1:3) = [ ...
         10.0,  5.0; ...
         11.0,  5.0; ...
         10.5, 15.0 ]';
    elseif ( i == 4 )
      t(1:dim_num,1:3) = [ ...
         10.0,  5.0; ...
         11.0,  5.0; ...
         20.0,   7.0 ]';
    end

    r8mat_print ( dim_num, 3, t, '  Triangle vertices ( columns )' );

    center = triangle_circumcenter_2d ( t );

    r8vec_print ( dim_num, center, '  Circumcenter :' );

    center = triangle_circumcenter_2d_2 ( t );

    r8vec_print ( dim_num, center, '  Circumcenter2:' );

  end

  return
end
