import Kriging
import Test

spherical_1 = Kriging.sphericalvariogram(0, 1, 10, 3.14)
spherical_2 = Kriging.sphericalvariogram(1, 1, 10, 3.14)
spherical_3 = Kriging.sphericalvariogram(20, 1, 10, 3.14)

exponential_1 = Kriging.exponentialvariogram(0, 1, 10, 3.14)
exponential_2 = Kriging.exponentialvariogram(1, 1, 10, 3.14)

gaussian_1 = Kriging.gaussianvariogram(0, 1, 10, 3.14)
gaussian_2 = Kriging.gaussianvariogram(1, 1, 10, 3.14)

mycov1 = h->Kriging.gaussiancov(h, 2, 300)
mycov2 = h->Kriging.expcov(h, 2, 300)
mycov3 = h->Kriging.sphericalcov(h, 2, 300)

x0 = [.5 .5; .49 .49; .01 .01; .99 1.; 0. 1.; 1. 0.]
xs = [0. 0.; 0. 1.; 1. 0.; 1. 1.; 0.500001 0.5]'
zs = [-20., .6, .4, 1., 20.]

krige_results_1 = Kriging.krige(x0, xs, zs, mycov1)
krige_results_2 = Kriging.krige(x0, xs, zs, mycov2)
krige_results_3 = Kriging.krige(x0, xs, zs, mycov3)

estimation_error_1 = Kriging.estimationerror(ones(size(xs, 2)), zs, xs, mycov1)
estimation_error_2 = Kriging.estimationerror(ones(size(xs, 2)), zs, xs, mycov2)
estimation_error_3 = Kriging.estimationerror(ones(size(xs, 2)), zs, xs, mycov3)

@Test.testset "Kriging" begin
	# Testing Kriging.sphericalvariogram()
	@Test.testset "Spherical Variogram" begin
		@Test.test isapprox(spherical_1, 0.0, atol=1e-6)
		@Test.test isapprox(spherical_2, 2.82007, atol=1e-6)
		@Test.test isapprox(spherical_3, 1, atol=1e-6)
	end

	# Testing Kriging.exponentialvariogram()
	@Test.testset "Exponential Variogram" begin
		@Test.test isapprox(exponential_1, 0.0, atol=1e-6)
		@Test.test isapprox(exponential_2, 3.069842455031493, atol=1e-6)
	end

	# Testing Kriging.gaussianvariogram()
	@Test.testset "Gaussian Variogram" begin
		@Test.test isapprox(gaussian_1, 0.0, atol=1e-6)
		@Test.test isapprox(gaussian_2, 3.13287854235668, atol=1e-6)
	end

	@Test.testset "Krige" begin
	   @Test.test isapprox(krige_results_1, [19.8891, 19.8891], atol=0.1)
	   @Test.test isapprox(krige_results_2, [19.4586, 19.4586], atol=0.1)
	   @Test.test isapprox(krige_results_3, [19.4586, 19.4586], atol=0.1)
	 end

	# Testing Kriging.estimationerror()
	@Test.testset "Estimation Error" begin
		@Test.test isapprox(estimation_error_1, 32.09281702460199, atol=1e-6)
		@Test.test isapprox(estimation_error_2, 33.19278009478499, atol=1e-6)
		@Test.test isapprox(estimation_error_3, 33.854178427022, atol=1e-6)
	end
end
:passed