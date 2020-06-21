function pn = cylinder_point_near_3d ( p1, p2, r, p )

%*****************************************************************************80
%
%% CYLINDER_POINT_NEAR_3D determines the nearest point on a cylinder to a point in 3D.
%
%  Discussion:
%
%    We are computing the nearest point on the SURFACE of the cylinder.
%
%    The surface of a (right) (finite) cylinder in 3D is defined by an axis,
%    which is the line segment from point P1 to P2, and a radius R.  The points
%    on the surface of the cylinder are:
%    * points at a distance R from the line through P1 and P2, and whose nearest
%      point on the line through P1 and P2 is strictly between P1 and P2,
%    PLUS
%    * points at a distance less than or equal to R from the line through P1
%      and P2, whose nearest point on the line through P1 and P2 is either
%      P1 or P2.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    14 March 2006
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
%    Input, real P(3), the point.
%
%    Output, real PN(3), the nearest point on the cylinder.
%
  dim_num = 3;

  axis(1:dim_num) = p2(1:dim_num) - p1(1:dim_num);
  axis_length = r8vec_length ( dim_num, axis );
  axis(1:dim_num) = axis(1:dim_num) / axis_length;

  axial_component = ( p(1:dim_num) - p1(1:dim_num) ) * axis';

  off_axis(1:dim_num) = p(1:dim_num) - p1(1:dim_num) ...
    - axial_component * axis(1:dim_num);

  off_axis_component = r8vec_length ( dim_num, off_axis );
%
%  Case 1: Below bottom cap.
%
  if ( axial_component <= 0.0 )

    if ( off_axis_component <= r )
      pn(1:dim_num) = p1(1:dim_num) + off_axis(1:dim_num);
    else
      pn(1:dim_num) = p1(1:dim_num) ...
        + ( r / off_axis_component ) * off_axis(1:dim_num);
    end
%
%  Case 2: between cylinder planes.
%
  elseif ( axial_component <= axis_length )

    if ( off_axis_component == 0.0 )
        
      off_axis = r8vec_any_normal ( dim_num, axis );
      pn(1:dim_num) = p(1:dim_num) + r * off_axis(1:dim_num);
      
    else
        
      distance = abs ( off_axis_component - r );
    
      pn(1:dim_num) = p1(1:dim_num) + axial_component * axis(1:dim_num) ...
        + ( r / off_axis_component ) * off_axis(1:dim_num);

      if ( off_axis_component < r )
        
        if ( axis_length - axial_component < distance )
          distance = axis_length - axial_component;
          pn(1:dim_num) = p2(1:dim_num) + off_axis(1:dim_num);
        end
      
        if ( axial_component < distance )
          distance = axial_component;
          pn(1:dim_num) = p1(1:dim_num) + off_axis(1:dim_num);
        end
      
      end

    end
%
%  Case 3: Above the top cap.
%
  elseif ( axis_length < axial_component )

    if ( off_axis_component <= r )
      pn(1:dim_num) = p2(1:dim_num) + off_axis(1:dim_num);
    else
      pn(1:dim_num) = p2(1:dim_num) ...
        + ( r / off_axis_component ) * off_axis(1:dim_num);
    end

  end

  return
end
