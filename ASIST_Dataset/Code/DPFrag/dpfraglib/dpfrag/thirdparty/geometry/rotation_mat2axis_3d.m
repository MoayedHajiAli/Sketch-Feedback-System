function [ axis, angle ] = rotation_mat2axis_3d ( a )

%*****************************************************************************80
%
%% ROTATION_MAT2AXIS_3D converts a rotation from matrix to axis format in 3D.
%
%  Discussion:
%
%    The computation is based on the fact that a rotation matrix must
%    have an eigenvector corresponding to the eigenvalue of 1, hence:
%
%      ( A - I ) * v = 0.
%
%    The eigenvector V is the axis of rotation.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    26 February 2005
%
%  Author:
%
%    John Burkardt
%
%  Reference:
%
%    Jack Kuipers,
%    Quaternions and Rotation Sequences,
%    Princeton, 1998.
%
%  Parameters:
%
%    Input, real A(3,3), the rotation matrix.
%
%    Output, real AXIS(3), the axis vector which remains
%    unchanged by the rotation.
%
%    Output, real ANGLE, the angular measurement of the
%    rotation about the axis, in radians.
%
  dim_num = 3;
%
%  Compute the normalized axis of rotation.
%
  axis(1) = a(3,2) - a(2,3);
  axis(2) = a(1,3) - a(3,1);
  axis(3) = a(2,1) - a(1,2);

  axis_norm = sqrt ( sum ( axis(1:dim_num).^2 ) );

  if ( axis_norm == 0.0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'ROTATION_MAT2AXIS_3D - Fatal error!\n' );
    fprintf ( 1, '  A is not a rotation matrix,\n' );
    fprintf ( 1, '  or there are multiple axes of rotation.\n' );
    error ( 'ROTATION_MAT2AXIS_3D - Fatal error!' );
  end

  axis(1:dim_num) = axis(1:dim_num) / axis_norm;
%
%  Find the angle.
%
  angle = arc_cosine ( 0.5 * ( a(1,1) + a(2,2) + a(3,3) - 1.0 ) );

  return
end
