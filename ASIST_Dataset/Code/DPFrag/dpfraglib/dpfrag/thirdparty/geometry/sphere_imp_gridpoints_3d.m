function [ npoint, p ] = sphere_imp_gridpoints_3d ( r, center, maxpoint, ...
  nlat, nlong )

%*****************************************************************************80
%
%% SPHERE_IMP_GRIDPOINTS_3D produces "grid points" on an implicit sphere in 3D.
%
%  Discussion:
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
%  Parameters:
%
%    Input, real R, the radius of the sphere.
%
%    Input, real CENTER(3), the center of the sphere.
%
%    Input, integer MAXPOINT, the maximum number of grid points, which
%    should be at least 2 + NLAT * NLONG.
%
%    Input, integer NLAT, NLONG, the number of latitude and longitude
%    lines to draw.  The latitudes do not include the North and South
%    poles, which will be included automatically, so NLAT = 5, for instance,
%    will result in points along 7 lines of latitude.
%
%    Output, integer NPOINT, the number of grid points.  The number of
%    grid points depends on N as follows:
%
%      NPOINT = 2 + NLAT * NLONG.
%
%    Output, real P(3,MAXPOINT), the grid points.
%
  dim_num = 3;
  npoint = 0;
%
%  The north pole.
%
  theta = 0.0;
  phi = 0.0;
  npoint = npoint + 1;
  if ( npoint <= maxpoint )
    p(1:dim_num,npoint) = sphere_imp_local2xyz_3d ( r, center, theta, phi )';
  end
%
%  Do each intermediate ring of latitude.
%
  for i = 1 : nlat

    phi = pi * i / ( nlat + 1 );
%
%  Along that ring of latitude, compute points at various longitudes.
%
    for j = 0 : nlong-1

      theta = 2.0 * pi * j / nlong;

      npoint = npoint + 1;
      if ( npoint <= maxpoint )
        p(1:dim_num,npoint) = sphere_imp_local2xyz_3d ( r, center, theta, phi )';
      end

    end
  end
%
%  The south pole.
%
  theta = 0.0;
  phi = pi;
  npoint = npoint + 1;
  if ( npoint <= maxpoint )
    p(1:dim_num,npoint) = sphere_imp_local2xyz_3d ( r, center, theta, phi )';
  end

  return
end
