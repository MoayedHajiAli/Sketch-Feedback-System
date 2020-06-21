function [ p, seed ] = cylinder_sample_3d ( p1, p2, r, n, seed )

%*****************************************************************************80
%
%% CYLINDER_SAMPLE_3D samples a cylinder in 3D.
%
%  Discussion:
%
%    We are sampling the interior of a right finite cylinder in 3D.
%
%    The interior of a (right) (finite) cylinder in 3D is defined by an axis,
%    which is the line segment from point P1 to P2, and a radius R.  The points
%    on or inside the cylinder are:
%    * points whose distance from the line through P1 and P2 is less than
%      or equal to R, and whose nearest point on the line through P1 and P2
%      lies (nonstrictly) between P1 and P2.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    17 October 2005
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real P1(3), P2(3), the first and last points
%    on the axis line of the cylinder.
%
%    Input, real R, the radius of the cylinder.
%
%    Input, integer N, the number of sample points to compute.
%
%    Input, integer SEED, the random number seed.
%
%    Input, real P(3,N), the sample points.
%
%    Output, integer SEED, the random number seed.
%
  dim_num = 3;
%
%  Compute the axis vector.
%
  axis(1:dim_num) = p2(1:dim_num) - p1(1:dim_num);
  axis_length = r8vec_length ( dim_num, axis );
  axis(1:dim_num) = axis(1:dim_num) / axis_length;
%
%  Compute vectors V2 and V3 that form an orthogonal triple with AXIS.
%
  [ v2, v3 ] = plane_normal_basis_3d ( p1, axis );
%
%  Assemble the randomized information.
%
  [ radius(1:n), seed ] = r8vec_uniform_01 ( n, seed );
  radius(1:n) = r * sqrt ( radius(1:n) );

  [ theta(1:n), seed ] = r8vec_uniform_01 ( n, seed );
  theta(1:n) = 2.0 * pi * theta(1:n);

  [ z(1:n), seed ] = r8vec_uniform_01 ( n, seed );
  z(1:n) = axis_length * z(1:n);

  for i = 1 : dim_num

    p(i,1:n) =                                       p1(i)   ...
              + z(1:n)                             * axis(i) ...
              + radius(1:n) .* cos ( theta(1:n) )   * v2(i)    ...
              + radius(1:n) .* sin ( theta(1:n) )   * v3(i);

  end

  return
end
