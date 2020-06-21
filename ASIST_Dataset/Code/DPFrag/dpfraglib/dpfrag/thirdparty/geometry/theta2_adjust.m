function [ theta1, theta2 ] = theta2_adjust ( theta1, theta2 )

%*****************************************************************************80
%
%% THETA2_ADJUST adjusts the theta coordinates of two points,
%
%  Discussion:
%
%    The THETA coordinate may be measured on a circle or sphere.  The
%    important thing is that it runs from 0 to 2 PI, and that it "wraps
%    around".  This means that two points may be close, while having
%    THETA coordinates that seem to differ by a great deal.  This program
%    is given a pair of THETA coordinates, and considers adjusting
%    one of them, by adding 2*PI, so that the range between
%    the large and small THETA coordinates is minimized.
%
%    This operation can be useful if, for instance, you have the THETA
%    coordinates of two points on a circle or sphere, and and you want
%    to generate intermediate points (by computing interpolated values
%    of THETA).  The values of THETA associated with the points must not 
%    have a "hiccup" or discontinuity in them, otherwise the interpolation 
%    will be ruined.
%
%    It should always be possible to adjust the THETA's so that the
%    range is at most PI.
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
%    Input, real THETA1, THETA2, two theta measurements.  It is assumed that
%    the values are between 0 and 2*PI (or actually, simply that they lie
%    within some interval of length at most 2*PI). 
%
%    Output, real THETA1, THETA2, one of the values may have been increased 
%    by 2*PI, to minimize the difference between the minimum and maximum 
%    values of THETA.
%
  if ( theta1 <= theta2 )

    if ( theta1 + 2.0 * pi - theta2 < theta2 - theta1 )
      theta1 = theta1 + 2.0 * pi;
    end

  elseif ( theta2 <= theta1 )

    if ( theta2 + 2.0 * pi - theta1 < theta1 - theta2 )
      theta2 = theta2 + 2.0 * pi;
    end

  end

  return
end
