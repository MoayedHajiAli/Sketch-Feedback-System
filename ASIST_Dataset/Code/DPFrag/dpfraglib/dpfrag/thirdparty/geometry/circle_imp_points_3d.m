function p = circle_imp_points_3d ( r, pc, nc, n )

%*****************************************************************************80
%
%% CIRCLE_IMP_POINTS_3D returns points on an implicit circle in 3D.
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
%    Input, real NC(3), a nonzero vector that is normal to
%    the plane of the circle.  It is customary, but not necessary,
%    that this vector have unit norm.
%
%    Input, integer N, the number of points desired.
%    N must be at least 1.
%
%    Output, real P(3,N), the coordinates of points
%    on the circle.
%
  dim_num = 3;
%
%  Get two unit vectors N1 and N2 which are orthogonal to each other,
%  and to NC.
%
  [ n1, n2 ] = plane_normal_basis_3d ( pc, nc );
%
%  Rotate R units away from PC in the plane of N1 and N2.
%
  for i = 1 : n

    theta = ( 2.0 * pi * ( i - 1 ) ) / n;

    p(1:dim_num,i) = ( pc(1:dim_num) ...
      + r * ( cos ( theta ) * n1(1:dim_num) ...
            + sin ( theta ) * n2(1:dim_num) ) )';

  end

  return
end
