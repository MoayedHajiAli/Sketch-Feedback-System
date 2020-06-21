function geometry_test049 ( )

%*****************************************************************************80
%
%% TEST049 tests PARALLELOGRAM_CONTAINS_POINT_3D.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    08 April 2009
%
%  Author:
%
%    John Burkardt
%
  dim_num = 3;
  ntest = 5;
%
%  In, Out, Out, Out, Out
%
  ptest(1:dim_num,1:ntest) = [ ...
    1.0,  1.0,  0.5; ...
    3.0,  3.0,  0.0; ...
    0.5,  0.5, -0.1; ...
    0.1,  0.1,  0.5; ...
    1.5,  1.6,  0.5 ]';

  p1(1:dim_num) = [ 0.0, 0.0, 0.0 ];
  p2(1:dim_num) = [ 2.0, 2.0, 0.0 ];
  p3(1:dim_num) = [ 1.0, 1.0, 1.0 ];

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST049\n' );
  fprintf ( 1, '  PARALLELOGRAM_CONTAINS_POINT_3D determines if a point\n' );
  fprintf ( 1, '    is within a parallelogram in 3D.\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '           P           Inside?\n' );
  fprintf ( 1, '\n' );

  for j = 1 : ntest

    p(1:dim_num) = ptest(1:dim_num,j);

    inside = parallelogram_contains_point_3d ( p1, p2, p3, p );

    fprintf ( 1, '  %12f  %12f  %12f  %d\n', p(1:dim_num), inside );

  end

  return
end
