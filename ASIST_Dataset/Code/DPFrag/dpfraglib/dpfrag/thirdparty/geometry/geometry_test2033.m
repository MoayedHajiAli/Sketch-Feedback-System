function geometry_test2033 ( )

%*****************************************************************************80
%
%% TEST2033 tests TETRAHEDRON_VOLUME_3D;
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    14 February 2003
%
%  Author:
%
%    John Burkardt
%
  dim_num = 3;

  tetra = [ ...
     0.000000,  0.942809, -0.333333; ...
    -0.816496, -0.816496, -0.333333; ...
     0.816496, -0.816496, -0.333333; ...
     0.000000,  0.000000,  1.000000 ]';

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST2033\n' );
  fprintf ( 1, '  For a tetrahedron in 3D,\n' );
  fprintf ( 1, '  TETRAHEDRON_VOLUME_3D computes the volume;\n' );

  r8mat_transpose_print ( dim_num, 4, tetra, '  Tetrahedron vertices' );

  volume = tetrahedron_volume_3d ( tetra );

  fprintf ( 1, '\n' );
  fprintf ( 1, '  Volume = %f\n', volume );

  return
end
