function diameter = triangle_diameter_2d ( t )

%*****************************************************************************80
%
%% TRIANGLE_DIAMETER_2D computes the diameter of a triangle in 2D.
%
%  Discussion:
%
%    The diameter of a triangle is the diameter of the smallest circle
%    that can be drawn around the triangle.  At least two of the vertices
%    of the triangle will intersect the circle, but not necessarily
%    all three!
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    30 January 2005
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real T(2,3), the triangle vertices.
%
%    Output, real DIAMETER, the diameter of the triangle.
%
  dim_num = 2;
%
%  Compute the squared length of each side.
%
  asq = sum ( t(1:dim_num,1) - t(1:dim_num,2) ).^2;
  bsq = sum ( t(1:dim_num,2) - t(1:dim_num,3) ).^2;
  csq = sum ( t(1:dim_num,3) - t(1:dim_num,1) ).^2;
%
%  Take care of a zero side.
%
  if ( asq == 0.0 )
    diameter = sqrt ( bsq );
    return
  elseif ( bsq == 0.0 )
    diameter = sqrt ( csq );
    return
  elseif ( csq == 0.0 )
    diameter = sqrt ( asq );
    return
  end
%
%  Make ASQ the largest.
%
  if ( asq < bsq )
    temp = asq;
    asq = bsq;
    bsq = temp;
  end

  if ( asq < csq )
    temp = asq;
    asq = csq;
    csq = temp;
  end
%
%  If ASQ is very large...
%
  if ( bsq + csq < asq )

    diameter = sqrt ( asq );

  else

    a = sqrt ( asq );
    b = sqrt ( bsq );
    c = sqrt ( csq );

    diameter = 2.0 * a * b * c / sqrt ( ( a + b + c ) * ( - a + b + c ) ...
      * ( a - b + c ) * ( a + b - c ) );

  end

  return
end
