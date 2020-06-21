function [ theta1, theta2, theta3 ] = theta3_adjust ( theta1, theta2, theta3 )

%*****************************************************************************80
%
%% THETA3_ADJUST adjusts the theta coordinates of three points,
%
%  Discussion:
%
%    The THETA coordinate may be measured on a circle or sphere.  The
%    important thing is that it runs from 0 to 2 PI, and that it "wraps
%    around".  This means that two points may be close, while having
%    THETA coordinates that seem to differ by a great deal.  This program
%    is given a set of three THETA coordinates, and tries to adjust
%    them, by adding 2*PI to some of them, so that the range between
%    the largest and smallest THETA coordinate is minimized.
%
%    This operation can be useful if, for instance, you have the THETA
%    coordinates of the three vertices of a spherical triangle, and
%    are trying to determine points inside the triangle by interpolation.
%    The values of THETA associated with the points must not have a
%    "hiccup" or discontinuity in them, otherwise the interpolation will
%    be ruined.
%
%    It should always be possible to adjust the THETA's so that the
%    range is at most 4 * PI / 3.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    22 July 2007
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input/output, real THETA1, THETA2, THETA3, three
%    theta measurements.  On input, it is assumed that the values
%    are all between 0 and 2*PI (or actually, simply that they lie
%    within some interval of length at most 2*PI). On output, some of
%    the values may have been increased by 2*PI, to minimize the
%    difference between the minimum and maximum values of THETA.
%
  if ( theta1 <= theta2 & theta2 <= theta3 )

    r1 = theta3            - theta1;
    r2 = theta1 + 2.0 * pi - theta2;
    r3 = theta2 + 2.0 * pi - theta3;

    if ( r2 < r1 & r2 < r3 )
      theta1 = theta1 + 2.0 * pi;
    elseif ( r3 < r1 & r3 < r2 )
      theta1 = theta1 + 2.0 * pi;
      theta2 = theta2 + 2.0 * pi;
    end

  elseif ( theta1 <= theta3 & theta3 <= theta2 )

    r1 = theta2            - theta1;
    r2 = theta1 + 2.0 * pi - theta3;
    r3 = theta3 + 2.0 * pi - theta2;

    if ( r2 < r1 & r2 < r3 )
      theta1 = theta1 + 2.0 * pi;
    elseif ( r3 < r1 & r3 < r2 )
      theta1 = theta1 + 2.0 * pi;
      theta3 = theta3 + 2.0 * pi;
    end

  elseif ( theta2 <= theta1 & theta1 <= theta3 )

    r1 = theta3            - theta2;
    r2 = theta2 + 2.0 * pi - theta1;
    r3 = theta1 + 2.0 * pi - theta3;

    if ( r2 < r1 & r2 < r3 )
      theta2 = theta2 + 2.0 * pi;
    elseif ( r3 < r1 & r3 < r2 )
      theta2 = theta2 + 2.0 * pi;
      theta1 = theta1 + 2.0 * pi;
    end

  elseif ( theta2 <= theta3 & theta3 <= theta1 )

    r1 = theta1            - theta2;
    r2 = theta2 + 2.0 * pi - theta3;
    r3 = theta3 + 2.0 * pi - theta1;

    if ( r2 < r1 & r2 < r3 )
      theta2 = theta2 + 2.0 * pi;
    elseif ( r3 < r1 & r3 < r2 )
      theta2 = theta2 + 2.0 * pi;
      theta3 = theta3 + 2.0 * pi;
    end

  elseif ( theta3 <= theta1 & theta1 <= theta2 )

    r1 = theta2            - theta3;
    r2 = theta3 + 2.0 * pi - theta1;
    r3 = theta1 + 2.0 * pi - theta2;

    if ( r2 < r1 & r2 < r3 )
      theta3 = theta3 + 2.0 * pi;
    elseif ( r3 < r1 & r3 < r2 )
      theta3 = theta3 + 2.0 * pi;
      theta1 = theta1 + 2.0 * pi;
    end

  elseif ( theta3 <= theta2 & theta2 <= theta1 )

    r1 = theta1            - theta3;
    r2 = theta3 + 2.0 * pi - theta2;
    r3 = theta2 + 2.0 * pi - theta1;

    if ( r2 < r1 & r2 < r3 )
      theta3 = theta3 + 2.0 * pi;
    elseif ( r3 < r1 & r3 < r2 )
      theta3 = theta3 + 2.0 * pi;
      theta2 = theta2 + 2.0 * pi;
    end

  end

  return
end

