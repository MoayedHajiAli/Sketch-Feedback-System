function det = r8mat_det_4d ( a )

%*****************************************************************************80
%
%% R8MAT_DET_4D computes the determinant of a 4 by 4 matrix.
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
%    Input, real A(4,4), the matrix whose determinant is desired.
%
%    Output, real DET, the determinant of the matrix.
%
  det = ...
      a(1,1) * ( ...
        a(2,2) * ( a(3,3) * a(4,4) - a(3,4) * a(4,3) ) ...
      - a(2,3) * ( a(3,2) * a(4,4) - a(3,4) * a(4,2) ) ...
      + a(2,4) * ( a(3,2) * a(4,3) - a(3,3) * a(4,2) ) ) ...
    - a(1,2) * ( ...
        a(2,1) * ( a(3,3) * a(4,4) - a(3,4) * a(4,3) ) ...
      - a(2,3) * ( a(3,1) * a(4,4) - a(3,4) * a(4,1) ) ...
      + a(2,4) * ( a(3,1) * a(4,3) - a(3,3) * a(4,1) ) ) ...
    + a(1,3) * ( ...
        a(2,1) * ( a(3,2) * a(4,4) - a(3,4) * a(4,2) ) ...
      - a(2,2) * ( a(3,1) * a(4,4) - a(3,4) * a(4,1) ) ...
      + a(2,4) * ( a(3,1) * a(4,2) - a(3,2) * a(4,1) ) ) ...
    - a(1,4) * ( ...
        a(2,1) * ( a(3,2) * a(4,3) - a(3,3) * a(4,2) ) ...
      - a(2,2) * ( a(3,1) * a(4,3) - a(3,3) * a(4,1) ) ...
      + a(2,3) * ( a(3,1) * a(4,2) - a(3,2) * a(4,1) ) );

  return
end
