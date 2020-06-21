function inside = parallelogram_contains_point_3d ( p1, p2, p3, p )

%*****************************************************************************80
%
%% PARALLELOGRAM_CONTAINS_POINT_3D determines if a point is inside a parallelogram in 3D.
%
%  Discussion:
%
%       P2..............
%       /              .
%      /              .
%     /              .
%    P1----------->P3
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    10 August 2005
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real P1(3), P2(3), P3(3), 
%    the vertices of the parallelogram.
%
%    Input, real P(3), the point to be checked.
%
%    Output, logical INSIDE, is TRUE if P is inside the
%    parallelogram, or on its boundary.
%    A slight amount of leeway is allowed for error, since a three
%    dimensional point may lie exactly in the plane of the parallelogram,
%    and yet be computationally slightly outside it.
%
  dim_num = 3;
  tol = 0.00001;
%
%  Compute P12, the vector normal to V1 = P2-P1 and V2 = P3-P1.
%
  p12(1) = ( p2(2) - p1(2) ) * ( p3(3) - p1(3) ) ...
         - ( p2(3) - p1(3) ) * ( p3(2) - p1(2) );
  p12(2) = ( p2(3) - p1(3) ) * ( p3(1) - p1(1) ) ...
         - ( p2(1) - p1(1) ) * ( p3(3) - p1(3) );
  p12(3) = ( p2(1) - p1(1) ) * ( p3(2) - p1(2) ) ...
         - ( p2(2) - p1(2) ) * ( p3(1) - p1(1) );
%
%  If the component of V = P-P1 in the P12 direction is too large,
%  then it does not lie in the parallelogram.
%
  dot = ( p(1:dim_num) - p1(1:dim_num) ) * p12(1:dim_num)';

  v = sqrt ( sum ( ( p2(1:dim_num) - p(1:dim_num) ).^2 ) );

  if ( tol * ( 1.0 + v ) < abs ( dot ) )
    inside = 0;
    return
  end
%
%  Compute P23, the vector normal to V2 and V3, by the cross product formula.
%
  p23(1) = ( p3(2) - p1(2) ) * p12(3) - ( p3(3) - p1(3) ) * p12(2);
  p23(2) = ( p3(3) - p1(3) ) * p12(1) - ( p3(1) - p1(1) ) * p12(3);
  p23(3) = ( p3(1) - p1(1) ) * p12(2) - ( p3(2) - p1(2) ) * p12(1);
%
%  Compute ALPHA = ( V dot P23 ) / ( V1 dot P23 )
%
  dott = ( p(1:dim_num)  - p1(1:dim_num) ) * p23(1:dim_num)';
  dotb = ( p2(1:dim_num) - p1(1:dim_num) ) * p23(1:dim_num)';

  if ( dotb < 0.0 )
    dott = -dott;
    dotb = -dotb;
  end

  if ( dott < 0.0 | dotb < dott )
    inside = 0;
    return
  end
%
%  Compute V31, the vector normal to P12 and V1, by the cross product formula.
%
  p31(1) = p12(2) * ( p2(3) - p1(3) ) - p12(3) * ( p2(1) - p1(2) );
  p31(2) = p12(3) * ( p2(1) - p1(1) ) - p12(1) * ( p2(3) - p1(3) );
  p31(3) = p12(1) * ( p2(2) - p1(2) ) - p12(2) * ( p2(1) - p1(1) );
%
%  Compute BETA = ( V dot V31 ) / ( V2 dot V31 )
%
  dott = ( p(1:dim_num)  - p1(1:dim_num) ) * p31(1:dim_num)';
  dotb = ( p3(1:dim_num) - p1(1:dim_num) ) * p31(1:dim_num)';

  if ( dotb < 0.0 )
    dott = -dott;
    dotb = -dotb;
  end

  if ( dott < 0.0 | dotb < dott )
    inside = 0;
    return
  end
%
%  V = ALPHA * V1 + BETA * V2, where both ALPHA and BETA are between
%  0 and 1.
%
  inside = 1;

  return
end
