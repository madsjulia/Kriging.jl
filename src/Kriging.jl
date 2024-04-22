module Kriging

import NearestNeighbors
import DocumentFunction
import LinearAlgebra
import Random

"Test Kriging"
function test()
	include(joinpath(Base.pkgdir(Kriging), "test", "runtests.jl"))
end

"""
Gaussian spatial covariance function

$(DocumentFunction.documentfunction(gaussiancov;
argtext=Dict("h"=>"separation distance",
			"maxcov"=>"maximum covariance",
			"scale"=>"scale",
			"nugget"=>"nugget")))

Returns:

- covariance
"""
gaussiancov(h::Number, maxcov::Number, scale::Number, nugget=Number=0.) = maxcov * exp(-(h * h) / (scale * scale)) + (h > 0 ? 0. : nugget)

"""
Exponential spatial covariance function

$(DocumentFunction.documentfunction(expcov;
argtext=Dict("h"=>"separation distance",
			"maxcov"=>"maximum covariance",
			"scale"=>"scale",
			"nugget"=>"nugget")))

Returns:

- covariance
"""
expcov(h::Number, maxcov::Number, scale::Number, nugget::Number=0.) = maxcov * exp(-h / scale) + (h > 0 ? 0. : nugget)

"""
Spherical spatial covariance function

$(DocumentFunction.documentfunction(sphericalcov;
argtext=Dict("h"=>"separation distance",
			"maxcov"=>"max covariance",
			"scale"=>"scale",
			"nugget"=>"nugget")))

Returns:

- covariance
"""
sphericalcov(h::Number, maxcov::Number, scale::Number, nugget::Number=0.) = (h <= scale ? maxcov * (1 - 1.5 * h / (scale) + .5 * (h / scale) ^ 3) : 0.) + (h > 0 ? 0. : nugget)
"""
Spherical variogram

$(DocumentFunction.documentfunction(sphericalvariogram;
argtext=Dict("h"=>"separation distance",
			"sill"=>"sill",
			"range"=>"range",
			"nugget"=>"nugget")))

Returns:

- Spherical variogram
"""
function sphericalvariogram(h::Number, sill::Number, range::Number, nugget::Number=0.)
	if h == 0.
		return 0.
	elseif h < range
		return (sill - nugget) * ((3 * h / (2 * range) - h ^ 3 / (2 * range ^ 3))) + nugget
	else
		return sill
	end
end

"""
Exponential variogram

$(DocumentFunction.documentfunction(exponentialvariogram;
argtext=Dict("h"=>"separation distance",
			"sill"=>"sill",
			"range"=>"range",
			"nugget"=>"nugget")))

Returns:

- Exponential variogram
"""
function exponentialvariogram(h::Number, sill::Number, range::Number, nugget::Number=0.)
	if h == 0.
		return 0.
	else
		return (sill - nugget) * (1 - exp(-h / (3 * range))) + nugget
	end
end

"""
Gaussian variogram

$(DocumentFunction.documentfunction(gaussianvariogram;
argtext=Dict("h"=>"separation distance",
			"sill"=>"sill",
			"range"=>"range",
			"nugget"=>"nugget")))

Returns:

- Gaussian variogram
"""
function gaussianvariogram(h::Number, sill::Number, range::Number, nugget::Number)
	if h == 0.
		return 0.
	else
		return (sill - nugget) * (1 - exp(-h * h / (3 * range * range))) + nugget
	end
end

function distance(a::AbstractArray, b::AbstractArray, pow::Number)
	result = 0.0
	for i in eachindex(a)
		result += abs.(a[i] .- b[i]) .^ pow
	end
	return result
end

function distsquared(a::AbstractArray, b::AbstractArray)
	result = 0.0
	for i in eachindex(a)
		result += abs.(a[i] .- b[i]) .^ 2
	end
	return result
end

function inversedistance(x0mat::AbstractMatrix, X::AbstractMatrix, Z::AbstractVector, f::Function; pow::Number=2, cutoff::Number=0)
	v = inversedistance(x0mat, X, Z, pow; cutoff=cutoff)
	return v, []
end

function inversedistance(x0mat::AbstractMatrix, X::AbstractMatrix, Z::AbstractVector, pow::Number=2; cutoff::Number=0)
	@assert size(X, 2) == length(Z) "Number of data points ($(size(X, 2))) and observations ($(length(Z))) do not match!"
	@assert size(x0mat, 1) == size(X, 1) "Dimensions of data points ($(size(x0mat, 1))) and observations ($(size(X, 1))) do not match!"
	iz = .!isnan.(Z)
	Xn = X[:,iz]
	result = Array{Float64}(undef, size(x0mat, 2))
	weights = Array{Float64}(undef, size(Xn, 2))
	for i in axes(x0mat, 2)
		for j in axes(Xn, 2)
			weights[j] = inv.(distance(x0mat[:, i], Xn[:, j], pow))
		end
		sw = sum(weights)
		if sw > cutoff
			result[i] = LinearAlgebra.dot(weights, Z[iz]) / sw
		else
			result[i] = NaN
		end
	end
	return result
end

"""
Simple Kriging

$(DocumentFunction.documentfunction(simplekrige;
argtext=Dict("x0mat"=>"point coordinates at which to obtain kriging estimates",
			"X"=>"coordinates of the observation (conditioning) data",
			"Z"=>"values for the observation (conditioning) data",
			"cov"=>"spatial covariance function")))

Returns:

- kriging estimates at `x0mat`
"""
function simplekrige(mu, x0mat::AbstractMatrix, X::AbstractMatrix, Z::AbstractVector, cov::Function)
	@assert size(X, 2) == length(Z) "Number of data points ($(size(X, 2))) and observations ($(length(Z))) do not match!"
	@assert size(x0mat, 2) == size(X, 1) "Dimensions of data points ($(size(x0mat, 2))) and observations ($(size(X, 1))) do not match!"
	result = fill(mu, size(x0mat, 2))
	resultvariance = fill(cov(0), size(x0mat, 2))
	covmat = getcovmat(X, cov)
	pinvcovmat = pinv(covmat)
	covvec = Array{Float64}(undef, size(X, 2))
	x = Array{Float64}(undef, size(X, 2))
	for i in axes(x0mat, 2)
		getcovvec!(covvec, x0mat[:, i], X, cov)
		A_mul_B!(x, pinvcovmat, covvec)
		for j in axes(X, 2)
			result[i] += (Z[j] - mu) * x[j]
			resultvariance[i] -= covvec[j] * x[j]
		end
	end
	return result, resultvariance
end

"""
Ordinary Kriging

$(DocumentFunction.documentfunction(krige;
argtext=Dict("x0mat"=>"point coordinates at which to obtain kriging estimates",
			"X"=>"coordinates of the observation (conditioning) data",
			"Z"=>"values for the observation (conditioning) data",
			"cov"=>"spatial covariance function")))

Returns:

- kriging estimates at `x0mat`
"""
function krige(x0mat::AbstractMatrix, X::AbstractMatrix, Z::AbstractVector, cov::Function; kw...)
	return first(krigevariance(x0mat, X, Z, cov; kw...))
end

"""
Ordinary Kriging plus variance

$(DocumentFunction.documentfunction(krige;
argtext=Dict("x0mat"=>"point coordinates at which to obtain kriging estimates",
			"X"=>"coordinates of the observation (conditioning) data",
			"Z"=>"values for the observation (conditioning) data",
			"cov"=>"spatial covariance function")))

Returns:

- kriging estimates at `x0mat`
- variance estimates at `x0mat`
"""
function krigevariance(x0mat::AbstractMatrix, X::AbstractMatrix, Z::AbstractVector, cov::Function; minwindow::Union{AbstractVector,Nothing}=nothing, maxwindow::Union{AbstractVector,Nothing}=nothing)
	@assert size(X, 2) == length(Z) "Number of data points ($(size(X, 2))) and observations ($(length(Z))) do not match!"
	@assert size(x0mat, 1) == size(X, 1) "Dimensions of data points ($(size(x0mat, 1))) and observations ($(size(X, 1))) do not match!"
	result = zeros(size(x0mat, 2))
	resultvariance = fill(cov(0), size(x0mat, 2))
	if !isnothing(minwindow) && !isnothing(maxwindow)
		if length(minwindow) != size(X, 1) || length(maxwindow) != size(X, 1)
			@error("minwindow and maxwindow must have the same length as the number of dimensions of the data!")
			result = fill(NaN, size(x0mat, 2))
			return result, resultvariance
		end
		if any(minwindow .> maxwindow)
			@error("minwindow must be less than or equal to maxwindow!")
			result = fill(NaN, size(x0mat, 2))
			return result, resultvariance
		end
		mask = vec(all(minwindow .< X .< maxwindow; dims=1))
		X = X[:, mask]
		Z = Z[mask]
	end

	covmat = getcovmat(X, cov)
	bigmat = [covmat ones(size(X, 2)); permutedims(ones(size(X, 2))) 0.]
	bigvec = Vector{Float64}(undef, size(X, 2) + 1)
	bigvec[end] = 1
	bigmatpinv = LinearAlgebra.pinv(bigmat)
	covvec = Vector{Float64}(undef, size(X, 2))
	x = Vector{Float64}(undef, size(X, 2) + 1)
	for i in axes(x0mat, 2)
		bigvec[1:end-1] = getcovvec!(covvec, x0mat[:, i], X, cov)
		bigvec[end] = 1
		LinearAlgebra.mul!(x, bigmatpinv, bigvec)
		for j in axes(X, 2)
			result[i] += Z[j] * x[j]
		end
		for j in eachindex(bigvec)
			resultvariance[i] -= bigvec[j] * x[j]
		end
	end
	return result, resultvariance
end

"""
Conditional Gaussian simulation

$(DocumentFunction.documentfunction(krige;
argtext=Dict("x0mat"=>"point coordinates at which to obtain kriging estimates",
			"X"=>"coordinates of the observation (conditioning) data",
			"Z"=>"values for the observation (conditioning) data",
			"cov"=>"spatial covariance function")))

Returns:

- conditional estimates at `x0mat`
"""
function condsim(x0mat::AbstractMatrix, X::AbstractMatrix, Z::AbstractVector, cov::Function, numneighbors, numobsneighbors=min(1000, size(X, 2)); neighborsearch=min(1000, size(x0mat, 2)))
	@assert size(X, 2) == length(Z) "Number of data points ($(size(X, 2))) and observations ($(length(Z))) do not match!"
	@assert size(x0mat, 1) == size(X, 1) "Dimensions of data points ($(size(x0mat, 1))) and observations ($(size(X, 1))) do not match!"
	nnindices, nnindices_obs = kdtree_indices(x0mat, X; numpredneighbors=neighborsearch, numobsneighbors=numobsneighbors)
	z0 = Vector{Float64}(undef, size(x0mat, 2))
	filledin = fill(false, size(x0mat, 2))
	perm = Random.randperm(size(x0mat, 2))
	for i in axes(x0mat, 2)
		thisx0 = reshape(x0mat[:, perm[i]], size(x0mat, 1), 1)
		neighbors = nnindices[perm[i]]
		obs_neighbors = nnindices_obs[perm[i]]
		numfilled = sum(filledin[neighbors])
		obs_size = min(numneighbors, numfilled) + numobsneighbors
		bigX = Matrix{Float64}(undef, size(x0mat, 1), obs_size)
		bigZ = Vector{Float64}(undef, obs_size)
		bigX[:, 1:numobsneighbors] = X[:, obs_neighbors]
		bigZ[1:numobsneighbors] = Z[obs_neighbors]
		bigXcount = numobsneighbors + 1
		j = 1
		while j <= length(neighbors) && bigXcount <= size(bigX, 2)
			if filledin[neighbors[j]]
				bigX[:, bigXcount] = x0mat[:, neighbors[j]]
				bigZ[bigXcount] = z0[neighbors[j]]
				bigXcount += 1
			end
			j += 1
		end
		mu, var = krigevariance(thisx0, bigX, bigZ, cov)
		z0[perm[i]] = mu[1] + sqrt(max(0., var[1])) * randn()
		filledin[perm[i]] = true
	end
	return z0
end

function interpolate_neighborhood(x0mat::AbstractMatrix, X::AbstractMatrix, Z::AbstractVector; cov::Function=cov(h)=Kriging.expcov(h, 0.1, 3), numobsneighbors=min(1000, size(X, 2)), neighborsearch=min(1000, size(x0mat, 2)), interpolate=krigevariance, return_variance::Bool=false, cutoff::Number=0, kw...)
	@assert size(X, 2) == length(Z) "Number of data points ($(size(X, 2))) and observations ($(length(Z))) do not match!"
	@assert size(x0mat, 1) == size(X, 1) "Dimensions of data points ($(size(x0mat, 1))) and observations ($(size(X, 1))) do not match!"
	_, nnindices_obs = kdtree_indices(x0mat, X; numpredneighbors=neighborsearch, numobsneighbors=numobsneighbors, cutoff_obs=cutoff)
	z0 = Vector{Float64}(undef, size(x0mat, 2))
	if return_variance
		var0 = Vector{Float64}(undef, size(x0mat, 2))
	end
	for i in axes(x0mat, 2)
		obs_neighbors = nnindices_obs[i]
		mu, var = interpolate(reshape(x0mat[:, i], size(x0mat, 1), 1), X[:, obs_neighbors], Z[obs_neighbors], cov; kw...)
		z0[i] = mu[1]
		if return_variance
			var0[i] = var[1]
		end
	end
	if return_variance
		return z0, var0
	else
		return z0
	end
end

function kdtree_indices(x_pred::AbstractMatrix{T}, x_obs::AbstractMatrix=Matrix{T}(undef, 0, 0); numpredneighbors=min(1000, size(x_pred, 2)), numobsneighbors=min(1000, size(x_obs, 2)), cutoff_pred::Number=0, cutoff_obs::Number=0) where T <: Real
	if numpredneighbors > 0
		kdtree = NearestNeighbors.KDTree(x_pred)
		nnindices_pred, nndistances_pred = NearestNeighbors.knn(kdtree, x_pred, numpredneighbors, true)
		if cutoff_pred > 0
			for i in eachindex(nnindices_pred)
				nnindices_pred[i] = nnindices_pred[i][nndistances_pred[i] .<= cutoff_pred]
			end
		end
	else
		nnindices_pred = []
	end
	if numobsneighbors > 0 && size(x_obs, 2) > 0
		@assert size(x_pred, 1) == size(x_obs, 1) "Dimensions of data points ($(size(x_pred, 1))) and observations ($(size(x_obs, 1))) do not match!"
		obs_kdtree = NearestNeighbors.KDTree(x_obs)
		nnindices_obs, nndistances_obs = NearestNeighbors.knn(obs_kdtree, x_pred, numobsneighbors, true)
		if cutoff_obs > 0
			for i in eachindex(nnindices_obs)
				nnindices_obs[i] = nnindices_obs[i][nndistances_obs[i] .<= cutoff_obs]
			end
		end
	else
		nndistances_obs = []
	end
	return nnindices_pred, nnindices_obs
end

"""
Get spatial covariance matrix

$(DocumentFunction.documentfunction(getcovmat;
argtext=Dict("X"=>"matrix with coordinates of the data points (x or y)",
			"cov"=>"spatial covariance function")))

Returns:

- spatial covariance matrix
"""
function getcovmat(X::AbstractMatrix, cov::Function)
	covmat = Array{Float64}(undef, size(X, 2), size(X, 2))
	cov0 = cov(0)
	for i in axes(X, 2)
		covmat[i, i] = cov0
		for j = i + 1:size(X, 2)
			covmat[i, j] = cov(LinearAlgebra.norm(X[:, i] - X[:, j]))
			covmat[j, i] = covmat[i, j]
		end
	end
	return covmat
end

"""
Get spatial covariance vector

$(DocumentFunction.documentfunction(getcovvec!;
argtext=Dict("covvec"=>"spatial covariance vector",
			"x0"=>"vector with coordinates of the estimation points (x or y)",
			"X"=>"matrix with coordinates of the data points",
			"cov"=>"spatial covariance function")))

Returns:

- spatial covariance vector
"""
function getcovvec!(covvec, x0::AbstractVector, X::AbstractMatrix, cov::Function)
	for i in axes(X, 2)
		d = 0.
		for j in axes(X, 1)
			d += (X[j, i] - x0[j]) ^ 2
		end
		d = sqrt(d)
		covvec[i] = cov(d)
	end
	return covvec
end

function estimationerror(w::AbstractVector, x0::AbstractVector, X, cov::Function)
	covmat = getcovmat(X, cov)
	covvec = Vector{Float64}(undef, size(X, 2))
	getcovvec!(covvec, x0, X, cov)
	cov0 = cov(0.)
	return estimationerror(w, x0, X, covmat, covvec, cov0)
end

function estimationerror(w::AbstractVector, x0::AbstractVector, X, covmat, covvec::AbstractVector, cov0::Number)
	return cov0 + LinearAlgebra.dot(w, covmat * w) - 2 * LinearAlgebra.dot(w, covvec)
end

@doc """
Estimate kriging error

$(DocumentFunction.documentfunction(estimationerror;
argtext=Dict("w"=>"kriging weights",
			"x0"=>"estimated locations",
			"X"=>"observation matrix",
			"cov"=>"spatial covariance function",
			"covmat"=>"covariance matrix",
			"covvec"=>"covariance vector",
			"cov0"=>"zero-separation covariance")))

Returns:

- estimation kriging error
""" estimationerror

function getgridpoints(xs::Union{AbstractVector,AbstractRange}, ys::Union{AbstractVector,AbstractRange})
	gridxy = Matrix{Float64}(undef, 2, length(xs) * length(ys))
	local i = 1
	for x in xs
		for y in ys
			gridxy[1, i] = x
			gridxy[2, i] = y
			i += 1
		end
	end
	return gridxy
end
function getgridpoints(xs::Union{AbstractVector,AbstractRange}, ys::Union{AbstractVector,AbstractRange}, zs::Union{AbstractVector,AbstractRange})
	gridxyz = Matrix{Float64}(undef, 3, length(xs) * length(ys) * length(zs))
	local i = 1
	for x in xs
		for y in ys
			for z in zs
				gridxyz[1, i] = x
				gridxyz[2, i] = y
				gridxyz[3, i] = z
				i += 1
			end
		end
	end
	return gridxyz
end

@doc """
Get grid points

$(DocumentFunction.documentfunction(getgridpoints;
argtext=Dict("xs"=>"x-axis grid coordinates",
			"ys"=>"y-axis grid coordinates",
			"zs"=>"z-axis grid coordinates")))

Returns:

- grid points
""" getgridpoints

function putgridpoints(xs::Union{AbstractVector,AbstractRange}, ys::Union{AbstractVector,AbstractRange}, zs::AbstractVector)
	@assert length(xs) * length(ys) == length(zs) "Number of grid points ($(length(xs) * length(ys))) and values ($(length(zs))) do not match!"
	gridxy = Matrix{Float64}(undef, length(xs), length(ys))
	local k = 1
	for (i, _) in enumerate(xs)
		for (j, _) in enumerate(ys)
			gridxy[i, j] = zs[k]
			k += 1
		end
	end
	return gridxy
end

function grid2layers(obs::AbstractVector, xs::AbstractVector, ys::AbstractVector, zs::AbstractVector)
	layers = Array{Array{Float64, 2}}(undef, length(zs))
	for k in eachindex(zs)
		layers[k] = Array{Float64}(undef, length(xs), length(ys))
		for i in eachindex(xs)
			for j in eachindex(ys)
				layers[k][i, j] = obs[k + (j - 1) * length(zs) + (i - 1) * length(zs) * length(ys)]
			end
		end
	end
	return layers
end

end
