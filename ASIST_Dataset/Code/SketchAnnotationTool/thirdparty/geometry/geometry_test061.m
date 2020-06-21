function geometry_test061 ( )

%*****************************************************************************80
%
%% TEST061 tests PLANE_NORMAL_BASIS_3D.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    07 April 2009
%
%  Author:
%
%    John Burkardt
%
  dim_num = 3;
  test_num = 5;
  seed = 123456789;

  fprintf ( 1, '\n' );
  fprintf ( 1, 'TEST061\n' );
  fprintf ( 1, '  PLANE_NORMAL_BASIS_3D, given a plane in\n' );
  fprintf ( 1, '    point, normal form (P,N), finds two unit\n' );
  fprintf ( 1, '    vectors Q and R that "lie" in the plane\n' );
  fprintf ( 1, '    and are mutually orthogonal.\n' );

  for test = 1 : test_num

    pp = r8vec_uniform_01 ( dim_num, seed );
    normal = r8vec_uniform_01 ( dim_num, seed );

    [ pq, pr ] = plane_normal_basis_3d ( pp, normal );

    if ( test == 1 )
      fprintf ( 1, '\n' );
      fprintf ( 1, '  Data for test 1:\n' );
      fprintf ( 1, '\n' );
      r8vec_print ( dim_num, pp, '  Point PP:' );
      r8vec_print ( dim_num, normal, '  Normal vector N:' );
      r8vec_print ( dim_num, pq, '  Vector PQ:' );
      r8vec_print ( dim_num, pr, '  Vector PR:' );
    end 

    b(1,1) = normal(1:dim_num) * normal(1:dim_num)';
    b(1,2) = normal(1:dim_num) * pq(1:dim_num)';
    b(1,3) = normal(1:dim_num) * pr(1:dim_num)';

    b(2,1) = pq(1:dim_num) * normal(1:dim_num)';
    b(2,2) = pq(1:dim_num) * pq(1:dim_num)';
    b(2,3) = pq(1:dim_num) * pr(1:dim_num)';

    b(3,1) = pr(1:dim_num) * normal(1:dim_num)';
    b(3,2) = pr(1:dim_num) * pq(1:dim_num)';
    b(3,3) = pr(1:dim_num) * pr(1:dim_num)';

    r8mat_print ( 3, 3, b, '  Dot product matrix:' );

  end

  return
end
