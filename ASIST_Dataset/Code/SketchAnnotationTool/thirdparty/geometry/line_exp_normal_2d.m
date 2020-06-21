function normal = line_exp_normal_2d ( p1, p2 )

%*****************************************************************************80
%
%% LINE_EXP_NORMAL_2D computes a unit normal vector to a line in 2D.
%
%  Discussion:
%
%    The explicit form of a line in 2D is:
%
%      the line through the points P1 and P2.
%
%    The sign of the normal vector N is chosen so that the normal vector
%    points "to the left" of the direction of the line.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    19 October 2006
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real P1(2), P2(2), two points on the line.
%
%    Output, real NORMAL(2), a unit normal vector to the line.
%
  dim_num = 2;

  norm = sqrt ( ( p2(1) - p1(1) ).^2 + ( p2(2) - p1(2) ).^2 );

  if ( norm == 0.0 )
    normal(1:dim_num) = sqrt ( 2.0 );
    return
  end

  normal(1) = - ( p2(2) - p1(2) ) / norm;
  normal(2) =   ( p2(1) - p1(1) ) / norm;

  return
end
