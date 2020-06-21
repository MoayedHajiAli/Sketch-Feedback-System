function p = sphere_imp_spiralpoints_3d ( r, center, n )

%*****************************************************************************80
%
%% SPHERE_IMP_SPIRALPOINTS_3D produces spiral points on an implicit sphere in 3D.
%
%  Discussion:
%
%    The points should be arranged on the sphere in a pleasing design.
%
%    An implicit sphere in 3D satisfies the equation:
%
%      sum ( ( P(1:DIM_NUM) - CENTER(1:DIM_NUM) )**2 ) = R**2
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
%  Reference:
%
%    E B Saff and A B J Kuijlaars,
%    Distributing Many Points on a Sphere,
%    The Mathematical Intelligencer,
%    Volume 19, Number 1, 1997, pages 5-11.
%
%  Parameters:
%
%    Input, real R, the radius of the sphere.
%
%    Input, real CENTER(3), the center of the sphere.
%
%    Input, integer N, the number of points to create.
%
%    Output, real P(3,N), the grid points.
%
  dim_num = 3;

  for i = 1 : n

    cosphi = ( ( n - i     ) * ( -1.0 )   ...
             + (     i - 1 ) * ( +1.0 ) ) ...
             / ( n     - 1 );

    sinphi = sqrt ( 1.0 - cosphi^2 );

    if ( i == 1 | i == n )
      theta = 0.0;
    else
      theta = theta + 3.6 / ( sinphi * sqrt ( n ) );
      theta = mod ( theta, 2.0 * pi );
    end

    p(1,i) = center(1) + r * sinphi * cos ( theta );
    p(2,i) = center(2) + r * sinphi * sin ( theta );
    p(3,i) = center(3) + r * cosphi;

  end

  return
end
