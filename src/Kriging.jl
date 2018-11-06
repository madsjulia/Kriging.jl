__precompile__()

module Kriging

import NearestNeighbors
import DocumentFunction
import LinearAlgebra

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

function distance(a::AbstractArray, b::AbstractArray)
	result = 0.0
	for i = 1:length(a)
		result += (a[i] - b[i])
	end
	return result
end

function distsquared(a::AbstractArray, b::AbstractArray)
	result = 0.0
	for i = 1:length(a)
		result += (a[i] - b[i]) ^ 2
	end
	return result
end

function inversedistance(x0mat::AbstractMatrix, X::AbstractMatrix, Z::AbstractVector, pow::Number)
	result = Array{Float64}(size(x0mat, 2))
	weights = Array{Float64}(size(X, 2))
	for i = 1:size(x0mat, 2)
		for j = 1:size(X, 2)
			weights[j] = 1 / distance(x0mat[:, i], X[:, j]) ^ pow
		end
		result[i] = dot(weights, Z) / sum(weights)
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
	result = fill(mu, size(x0mat, 2))
	resultvariance = fill(cov(0), size(x0mat, 2))
	covmat = getcovmat(X, cov)
	pinvcovmat = pinv(covmat)
	covvec = Array{Float64}(size(X, 2))
	x = Array{Float64}(size(X, 2))
	for i = 1:size(x0mat, 2)
		getcovvec!(covvec, x0mat[:, i], X, cov)
		A_mul_B!(x, pinvcovmat, covvec)
		for j = 1:size(X, 2)
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
function krige(x0mat::AbstractMatrix, X, Z::AbstractVector, cov::Function)
	return krigevariance(x0mat, X, Z, cov)[1]
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
function krigevariance(x0mat::AbstractMatrix, X, Z::AbstractVector, cov::Function)
	if size(X, 2) != length(Z)
		error("number of points and observations don't match")
	end
	result = zeros(size(x0mat, 2))
	resultvariance = fill(cov(0), size(x0mat, 2))
	covmat = getcovmat(X, cov)
	bigmat = [covmat ones(size(X, 2)); ones(size(X, 2))' 0.]
	ws = Array{Float64}(undef, size(x0mat, 2))
	bigvec = Array{Float64}(undef, size(X, 2) + 1)
	bigvec[end] = 1
	bigmatpinv = LinearAlgebra.pinv(bigmat)
	covvec = Array{Float64}(undef, size(X, 2))
	x = Array{Float64}(undef, size(X, 2) + 1)
	for i = 1:size(x0mat, 2)
		bigvec[1:end-1] = getcovvec!(covvec, x0mat[:, i], X, cov)
		bigvec[end] = 1
		LinearAlgebra.mul!(x, bigmatpinv, bigvec)
		for j = 1:size(X, 2)
			result[i] += Z[j] * x[j]
		end
		for j = 1:length(bigvec)
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
function condsim(x0mat::AbstractMatrix, X::AbstractMatrix, Z::AbstractVector, cov::Function, numneighbors, numobsneighbors=length(Z); neighborsearch=min(1000, size(x0mat, 2)))
	kdtree = NearestNeighbors.KDTree(x0mat)
	nnindices, _ = NearestNeighbors.knn(kdtree, x0mat, neighborsearch, true)
	obs_kdtree = NearestNeighbors.KDTree(X)
	nnindices_obs, _ = NearestNeighbors.knn(obs_kdtree, x0mat, numobsneighbors, true)
	z0 = Array{Float64}(size(x0mat, 2))
	filledin = fill(false, size(x0mat, 2))
	perm = randperm(size(x0mat, 2))
	maxvar = 0.
	for i = 1:size(x0mat, 2)
		thisx0 = reshape(x0mat[:, perm[i]], size(x0mat, 1), 1)
		neighbors = nnindices[perm[i]]
		obs_neighbors = nnindices_obs[perm[i]]
		numfilled = sum(filledin[neighbors])
		bigX = Array{Float64}(size(x0mat, 1), min(numneighbors, numfilled) + numobsneighbors)
		bigZ = Array{Float64}(min(numneighbors, numfilled) + numobsneighbors)
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

"""
Get spatial covariance matrix

$(DocumentFunction.documentfunction(getcovmat;
argtext=Dict("X"=>"matrix with coordinates of the data points (x or y)",
			"cov"=>"spatial covariance function")))

Returns:

- spatial covariance matrix
"""
function getcovmat(X::AbstractMatrix, cov::Function)
	covmat = Array{Float64,2}(undef, size(X, 2), size(X, 2))
	cov0 = cov(0)
	for i = 1:size(X, 2)
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
function getcovvec!(covvec, x0::AbstractVector, X, cov::Function)
	for i = 1:size(X, 2)
		d = 0.
		for j = 1:size(X, 1)
			d += (X[j, i] - x0[j]) ^ 2
		end
		d = sqrt(d)
		covvec[i] = cov(d)
	end
	return covvec
end

function estimationerror(w::AbstractVector, x0::AbstractVector, X, cov::Function)
	covmat = getcovmat(X, cov)
	covvec = Array{Float64}(undef, size(X, 2))
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

function getgridpoints(xs::AbstractVector, ys::AbstractVector)
	gridxyz = Array{Float64}(2, length(xs) * length(ys))
	i = 1
	for x in xs
		for y in ys
			gridxyz[1, i] = x
			gridxyz[2, i] = y
			i += 1
		end
	end
	return gridxyz
end

function getgridpoints(xs::AbstractVector, ys::AbstractVector, zs::AbstractVector)
	gridxyz = Array{Float64}(3, length(xs) * length(ys) * length(zs))
	i = 1
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

function grid2layers(obs::AbstractVector, xs::AbstractVector, ys::AbstractVector, zs::AbstractVector)
	layers = Array{Array{Float64, 2}}(length(zs))
	for k = 1:length(zs)
		layers[k] = Array{Float64}(length(xs), length(ys))
		for i = 1:length(xs)
			for j = 1:length(ys)
				layers[k][i, j] = obs[k + (j - 1) * length(zs) + (i - 1) * length(zs) * length(ys)]
			end
		end
	end
	return layers
end

end
