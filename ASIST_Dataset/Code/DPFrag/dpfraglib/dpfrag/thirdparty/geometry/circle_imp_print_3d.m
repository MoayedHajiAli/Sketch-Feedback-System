function circle_imp_print_3d ( r, pc, nc, title )

%*****************************************************************************80
%
%% CIRCLE_IMP_PRINT_2D prints an implicit circle in 3D.
%
%  Discussion:
%
%    Points P on an implicit circle in 3D satisfy the equations:
%
%      ( P(1) - PC(1) )**2
%    + ( P(2) - PC(2) )**2
%    + ( P(3) - PC(3) )**2 = R**2
%
%    and
%
%      ( P(1) - PC(1) ) * NC(1)
%    + ( P(2) - PC(2) ) * NC(2)
%    + ( P(3) - PC(3) ) * NC(3) = 0
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    10 March 2006
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real R, the radius of the circle.
%
%    Input, real PC(3), the center of the circle.
%
%    Input, real NC(3), the normal vector to the circle.
%
%    Input, string TITLE, an optional title.
%
  dim_num = 3;

  if ( 0 < s_len_trim ( title ) )
    fprintf ( 1, '\n' );
    fprintf ( 1, '%s\n', title );
  end

  fprintf ( 1, '\n' );
  fprintf ( 1, '  Radius = %f\n', r );
  fprintf ( 1, '  Center = %f  %f  %f\n', pc(1:dim_num) );
  fprintf ( 1, '  Normal = %f  %f  %f\n', nc(1:dim_num) );

  return
end
