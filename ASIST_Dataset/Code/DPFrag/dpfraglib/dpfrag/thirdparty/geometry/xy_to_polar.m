function [ r, t ] = xy_to_polar ( xy )

%*****************************************************************************80
%
%% XY_TO_POLAR converts XY coordinates to polar coordinates.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    10 January 2007
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real XY(2), the Cartesian coordinates.
%
%    Output, real R, T, the radius and angle (in radians).
%
  r = sqrt ( xy(1) .* xy(1) + xy(2) .* xy(2) );

  if ( r == 0.0 )
    t = 0.0;
  else
    t = atan2 ( xy(2), xy(1) );
  end

  return
end
