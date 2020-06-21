function [ point_coord, face_order, face_point ] = soccer_shape_3d ( ...
  point_num, face_num, face_order_max )

%*****************************************************************************80
%
%% SOCCER_SHAPE_3D describes a truncated icosahedron in 3D.
%
%  Discussion:
%
%    The shape is a truncated icosahedron, which is the design used
%    on a soccer ball.  There are 12 pentagons and 20 hexagons.
%
%    Call SOCCER_SIZE_3D to get the values of POINT_NUM, FACE_NUM, and 
%    FACE_ORDER_MAX, so you can allocate space for the arrays.
%
%    For each face, the face list must be of length FACE_ORDER_MAX.
%    In cases where a face is of lower than maximum order (the
%    12 pentagons, in this case), the extra entries are listed as
%    "-1".
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
%  Reference:
%
%    http://mathworld.wolfram.com/TruncatedIcosahedron.html
%
%  Parameters:
%
%    Input, integer POINT_NUM, the number of points in the shape (60).
%
%    Input, integer FACE_NUM, the number of faces in the shape (32).
%
%    Input, integer FACE_ORDER_MAX, the maximum order of any face (6).
%
%    Output, real POINT_COORD(3,POINT_NUM), the vertices.
%
%    Output, integer FACE_ORDER(FACE_NUM), the number of vertices per face.
%
%    Output, integer FACE_POINT(FACE_ORDER_MAX,FACE_NUM); FACE_POINT(I,J)
%    contains the index of the I-th point in the J-th face.  The
%    points are listed in the counter-clockwise direction defined
%    by the outward normal at the face.
%
  dim_num = 3;
%
%  Set the point coordinates.
%
  point_coord(1:dim_num,1:point_num) = [ ...
       -1.00714,    0.153552,   0.067258; ...
       -0.960284,   0.0848813, -0.33629; ...
       -0.95172,   -0.153552,   0.33629; ...
       -0.860021,   0.529326,   0.150394; ...
       -0.858,     -0.290893,  -0.470806; ...
       -0.849436,  -0.529326,   0.201774; ...
       -0.802576,  -0.597996,  -0.201774; ...
       -0.7842,     0.418215,  -0.502561; ...
       -0.749174,  -0.0848813,  0.688458; ...
       -0.722234,   0.692896,  -0.201774; ...
       -0.657475,   0.597996,   0.502561; ...
       -0.602051,   0.290893,   0.771593; ...
       -0.583675,  -0.692896,   0.470806; ...
       -0.579632,  -0.333333,  -0.771593; ...
       -0.52171,   -0.418215,   0.771593; ...
       -0.505832,   0.375774,  -0.803348; ...
       -0.489955,  -0.830237,  -0.33629; ...
       -0.403548,   0.,        -0.937864; ...
       -0.381901,   0.925138,  -0.201774; ...
       -0.352168,  -0.666667,  -0.688458; ...
       -0.317142,   0.830237,   0.502561; ...
       -0.271054,  -0.925138,   0.33629; ...
       -0.227464,   0.333333,   0.937864; ...
       -0.224193,  -0.993808,  -0.067258; ...
       -0.179355,   0.993808,   0.150394; ...
       -0.165499,   0.608015,  -0.803348; ...
       -0.147123,  -0.375774,   0.937864; ...
       -0.103533,   0.882697,  -0.502561; ...
       -0.0513806,  0.666667,   0.771593; ...
        0.0000000,  0.,         1.021; ...
        0.0000000,  0.,        -1.021; ...
        0.0513806, -0.666667,  -0.771593; ...
        0.103533,  -0.882697,   0.502561; ...
        0.147123,   0.375774,  -0.937864; ...
        0.165499,  -0.608015,   0.803348; ...
        0.179355,  -0.993808,  -0.150394; ...
        0.224193,   0.993808,   0.067258; ...
        0.227464,  -0.333333,  -0.937864; ...
        0.271054,   0.925138,  -0.33629; ...
        0.317142,  -0.830237,  -0.502561; ...
        0.352168,   0.666667,   0.688458; ...
        0.381901,  -0.925138,   0.201774; ...
        0.403548,   0.,         0.937864; ...
        0.489955,   0.830237,   0.33629; ...
        0.505832,  -0.375774,   0.803348; ...
        0.521710,   0.418215,  -0.771593; ...
        0.579632,   0.333333,   0.771593; ...
        0.583675,   0.692896,  -0.470806; ...
        0.602051,  -0.290893,  -0.771593; ...
        0.657475,  -0.597996,  -0.502561; ...
        0.722234,  -0.692896,   0.201774; ...
        0.749174,   0.0848813, -0.688458; ...
        0.784200,  -0.418215,   0.502561; ...
        0.802576,   0.597996,   0.201774; ...
        0.849436,   0.529326,  -0.201774; ...
        0.858000,   0.290893,   0.470806; ...
        0.860021,  -0.529326,  -0.150394; ...
        0.951720,   0.153552,  -0.33629; ...
        0.960284,  -0.0848813,  0.33629; ...
        1.007140,  -0.153552,  -0.067258 ]';
%
%  Set the face orders.
%
  face_order(1:face_num) = [ ...
    6, 6, 5, 6, 5, 6, 5, 6, 6, 6, ...
    5, 6, 5, 6, 5, 6, 6, 6, 5, 6, ...
    5, 5, 6, 6, 6, 5, 6, 5, 6, 6, ...
    5, 6 ];
%
%  Set faces.
%
  face_point(1:face_order_max,1:face_num) = [ ...
       30, 43, 47, 41, 29, 23; ...
       30, 23, 12,  9, 15, 27; ...
       30, 27, 35, 45, 43, -1; ...
       43, 45, 53, 59, 56, 47; ...
       23, 29, 21, 11, 12, -1; ...
       27, 15, 13, 22, 33, 35; ...
       47, 56, 54, 44, 41, -1; ...
       45, 35, 33, 42, 51, 53; ...
       12, 11,  4,  1,  3,  9; ...
       29, 41, 44, 37, 25, 21; ...
       15,  9,  3,  6, 13, -1; ...
       56, 59, 60, 58, 55, 54; ...
       53, 51, 57, 60, 59, -1; ...
       11, 21, 25, 19, 10,  4; ...
       33, 22, 24, 36, 42, -1; ...
       13,  6,  7, 17, 24, 22; ...
       54, 55, 48, 39, 37, 44; ...
       51, 42, 36, 40, 50, 57; ...
        4, 10,  8,  2,  1, -1; ...
        3,  1,  2,  5,  7,  6; ...
       25, 37, 39, 28, 19, -1; ...
       55, 58, 52, 46, 48, -1; ...
       60, 57, 50, 49, 52, 58; ...
       10, 19, 28, 26, 16,  8; ...
       36, 24, 17, 20, 32, 40; ...
        7,  5, 14, 20, 17, -1; ...
       48, 46, 34, 26, 28, 39; ...
       50, 40, 32, 38, 49, -1; ...
        8, 16, 18, 14,  5,  2; ...
       46, 52, 49, 38, 31, 34; ...
       16, 26, 34, 31, 18, -1; ...
       32, 20, 14, 18, 31, 38 ]';

  return
end
