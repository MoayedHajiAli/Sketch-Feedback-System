function [ vran, seed ] = direction_pert_3d ( sigma, vbase, seed )

%*****************************************************************************80
%
%% DIRECTION_PERT_3D randomly perturbs a direction vector in 3D.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    09 February 2005
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real SIGMA, determines the strength of the
%    perturbation.
%    SIGMA <= 0 results in a completely random direction.
%    1 <= SIGMA results in VBASE.
%    0 < SIGMA < 1 results in a perturbation from VBASE, which is
%    large when SIGMA is near 0, and small when SIGMA is near 1.
%
%    Input, real VBASE(3), the base direction vector, which
%    should have unit norm.
%
%    Input, integer SEED, a seed for the random number generator.
%
%    Output, real VRAN(3), the perturbed vector, which will
%    have unit norm.
%
%    Output, integer SEED, a seed for the random number generator.
%
  dim_num = 3;
%
%  1 <= SIGMA, just use the base vector.
%
  if ( 1.0 <= sigma )

    vran(1:dim_num) = vbase(1:dim_num);

  elseif ( sigma <= 0.0 )

    [ vdot, seed ] = r8_uniform_01 ( seed );
    vdot = 2.0 * vdot - 1.0;

    phi = arc_cosine ( vdot );

    [ theta, seed ] = r8_uniform_01 ( seed );
    theta = 2.0 * pi * theta;

    vran(1) = cos ( theta ) * sin ( phi );
    vran(2) = sin ( theta ) * sin ( phi );
    vran(3) = cos ( phi );

  else

    phi = arc_cosine ( vbase(3) );
    theta = atan2 ( vbase(2), vbase(1) );
%
%  Pick VDOT, which must be between -1 and 1.  This represents
%  the dot product of the perturbed vector with the base vector.
%
%  R8_UNIFORM_01 returns a uniformly random value between 0 and 1.
%  The operations we perform on this quantity tend to bias it
%  out towards 1, as SIGMA grows from 0 to 1.
%
%  VDOT, in turn, is a value between -1 and 1, which, for large
%  SIGMA, we want biased towards 1.
%
    [ r, seed ] = r8_uniform_01 ( seed );
    x = exp ( ( 1.0 - sigma ) * log ( r ) );
    dphi = arc_cosine ( 2.0 * x - 1.0 );
%
%  Now we know enough to write down a vector that is rotated DPHI
%  from the base vector.
%
    v(1) = cos ( theta ) * sin ( phi + dphi );
    v(2) = sin ( theta ) * sin ( phi + dphi );
    v(3) = cos ( phi + dphi );
%
%  Pick a uniformly random rotation between 0 and 2 Pi around the
%  axis of the base vector.
%
    [ psi, seed ] = r8_uniform_01 ( seed );
    psi = 2.0 * pi * psi;
%
%  Carry out the rotation.
%
    vran = rotation_axis_vector_3d ( vbase, psi, v );

  end

  return
end
