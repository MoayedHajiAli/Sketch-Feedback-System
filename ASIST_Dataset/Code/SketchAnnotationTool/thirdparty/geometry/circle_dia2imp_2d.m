function [ r, center ] = circle_dia2imp_2d ( p1, p2 )

%*****************************************************************************80
%
%% CIRCLE_DIA2IMP_2D converts a diameter to an implicit circle in 2D.
%
%  Discussion:
%
%    The diameter form of a circle is:
%
%      P1 and P2 are the endpoints of a diameter.
%
%    The implicit form of a circle in 2D is:
%
%      ( X - CENTER(1) )**2 + ( Y - CENTER(2) )**2 = R**2
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    31 January 2005
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real P1(2), P2(2), two points that are the 
%    endpoints of a diameter of the circle.
%
%    Output, real R, the radius of the circle.
%
%    Output, real CENTER(2), the center of the circle.
%
  dim_num = 2;

  r = 0.5 * sqrt ( sum ( ( p2(1:dim_num) - p1(1:dim_num) ).^2 ) );

  center(1:dim_num) = 0.5 * ( p1(1:dim_num) + p2(1:dim_num) );

  return
end
