function pint = box_ray_int_2d ( p1, p2, pa, pb )

%*****************************************************************************80
%
%% BOX_RAY_INT_2D: intersection ( box, ray ) in 2D.
%
%  Discussion:
%
%    A box is assumed to be a rectangle with sides aligned on coordinate
%    axes.  It can be described by its low and high corner, P1 and P2:
%
%      points P so that P1(1:DIM_NUM) <= P(1:DIM_NUM) <= P2(1:DIM_NUM).
%
%    The origin of the ray is assumed to be inside the box.  This
%    guarantees that the ray will intersect the box in exactly one point.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    05 May 2005
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real P1(2), P2(2), the low and high corners of the box.
%
%    Input, real PA(2), the origin of the ray, which should be
%    inside the box.
%
%    Input, real PB(2), a second point on the ray.
%
%    Output, real PINT(2), a point on the box intersected 
%    by the ray.
%
  dim_num = 2;

  for side = 1 : 4

    if ( side == 1 )
      pd(1:2) = [ p1(1), p1(2) ];
      pc(1:2) = [ p2(1), p1(2) ];
    elseif ( side == 2 )
      pd(1:2) = [ p2(1), p1(2) ];
      pc(1:2) = [ p2(1), p2(2) ];
    elseif ( side == 3 )
      pd(1:2) = [ p2(1), p2(2) ];
      pc(1:2) = [ p1(1), p2(2) ];
    elseif ( side == 4 )
      pd(1:2) = [ p1(1), p2(2) ];
      pc(1:2) = [ p1(1), p1(2) ];
    end

    inside = angle_contains_point_2d ( pc, pa, pd, pb );

    if ( inside )
      [ ival, pint ] = lines_exp_int_2d ( pa, pb, pc, pd );
      if ( ival == 0 )
        fprintf ( 1, '\n' );
        fprintf ( 1, 'BOX_RAY_INT_2D - Fatal error!\n' );
        fprintf ( 1, '  Could not find a point of intersection.\n' );
        error ( 'BOX_RAY_INT_2D - Fatal error!' );
      end
      return
    end

  end

  fprintf ( 1, '\n' );
  fprintf ( 1, 'BOX_RAY_INT_2D - Fatal error!\n' );
  fprintf ( 1, '  Could not find a point of intersection.\n' );
  error ( 'BOX_RAY_INT_2D - Fatal error!' );
  
  return
end
