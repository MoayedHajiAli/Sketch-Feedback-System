function geometry_test211 ( )

%*****************************************************************************80
%
%% TEST211 tests TRIANGLE_ORIENTATION_2D.
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

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST211\n' );
  fprintf ( 1, '  TRIANGLE_ORIENTATION_2D determines orientation\n' );
  fprintf ( 1, '    of a triangle.\n' );

  t(1:dim_num,1:3) = [ ...
    4.0, 2.0; ...
    1.0, 5.0; ...
   -2.0, 2.0 ]';

  i = triangle_orientation_2d ( t );

  r8mat_transpose_print ( dim_num, 3, t, '  Triangle vertices:' );

  fprintf ( 1, '\n' );
  if ( i == 0 )
    fprintf ( 1, '  The points are counterclockwise.\n' );
  elseif ( i == 1 )
    fprintf ( 1, '  The points are clockwise.\n' );
  elseif ( i == 2 )
    fprintf ( 1, '  The points are colinear.\n' );
  elseif ( i == 3 )
    fprintf ( 1, '  The points are not distinct.\n' );
  else
    fprintf ( 1, '  The return value makes no sense.\n' );
  end

  t(1:dim_num,1:3) = [ ...
    1.0,  5.0; ...
    4.0,  2.0; ...
    1.0, -1.0 ]';

  i = triangle_orientation_2d ( t );

  r8mat_transpose_print ( dim_num, 3, t, '  Triangle vertices:' );

  fprintf ( 1, '\n' );
  if ( i == 0 )
    fprintf ( 1, '  The points are counterclockwise.\n' );
  elseif ( i == 1 )
    fprintf ( 1, '  The points are clockwise.\n' );
  elseif ( i == 2 )
    fprintf ( 1, '  The points are colinear.\n' );
  elseif ( i == 3 )
    fprintf ( 1, '  The points are not distinct.\n' );
  else
    fprintf ( 1, '  The return value makes no sense.\n' );
  end
%
%  Colinear points
%
  t(1:dim_num,1:3) = [ ...
    1.0, 5.0; ...
    2.0, 7.0; ...
    3.0, 9.0 ]';

  i = triangle_orientation_2d ( t );

  r8mat_transpose_print ( dim_num, 3, t, '  Triangle vertices:' );

  fprintf ( 1, '\n' );
  if ( i == 0 )
    fprintf ( 1, '  The points are counterclockwise.\n' );
  elseif ( i == 1 )
    fprintf ( 1, '  The points are clockwise.\n' );
  elseif ( i == 2 )
    fprintf ( 1, '  The points are colinear.\n' );
  elseif ( i == 3 )
    fprintf ( 1, '  The points are not distinct.\n' );
  else
    fprintf ( 1, '  The return value makes no sense.\n' );
  end
%
%  Nondistinct points.
%
  t(1:dim_num,1:3) = [ ...
    1.0, 5.0; ...
    4.0, 2.0; ...
    1.0, 5.0 ]';

  i = triangle_orientation_2d ( t );

  r8mat_transpose_print ( dim_num, 3, t, '  Triangle vertices:' );

  fprintf ( 1, '\n' );
  if ( i == 0 )
    fprintf ( 1, '  The points are counterclockwise.\n' );
  elseif ( i == 1 )
    fprintf ( 1, '  The points are clockwise.\n' );
  elseif ( i == 2 )
    fprintf ( 1, '  The points are colinear.\n' );
  elseif ( i == 3 )
    fprintf ( 1, '  The points are not distinct.\n' );
  else
    fprintf ( 1, '  The return value makes no sense.\n' );
  end

  return
end
