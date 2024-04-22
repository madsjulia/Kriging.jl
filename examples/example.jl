import Kriging
import PyPlot
import Random

cd(@__DIR__)

Random.seed!(0)
X = Float64.([3 7 1 5 9; 3 2 7 9 6])
Z = exp.(randn(size(X, 2)))
fig, ax = PyPlot.subplots()
for i in axes(X, 2)
	ax.plot([X[1, i]], [X[2, i]], "r.", ms=10)
	ax.text([X[1, i] + 0.1], [X[2, i] + 0.1], "$(round(Z[i]; sigdigits=2))", fontsize=16)
end
ax.plot([5], [5], "r.", ms=10)
ax.text([5.1], [5.1], "?", fontsize=16)
ax.axis("scaled")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
PyPlot.title("Data")
display(fig); println()
fig.savefig("data.pdf")
PyPlot.close(fig)

covfun(h) = Kriging.expcov(h, 0.1, 3)
xs = collect(range(0; stop=10, length=100))
ys = collect(range(0, stop=10, length=100))
x0mat = Kriging.getgridpoints(xs, ys)

krigedfield = Array{Float64}(undef, length(xs), length(ys))
@time for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
	krigedfield[i, j] = Kriging.krige(permutedims([x y]), X, Z, covfun)[1]
end

fig, ax = PyPlot.subplots()
cax = ax.imshow(permutedims(krigedfield), extent=[0, 10, 0, 10], origin="lower")
for i in axes(X, 2)
	ax.plot([X[1, i]], [X[2, i]], "r.", ms=10)
end
fig.colorbar(cax)
PyPlot.title("Kriging")
display(fig); println()
fig.savefig("kriging.pdf")
PyPlot.close(fig)

@time z0 = Kriging.interpolate_neighborhood(x0mat, X, Z; cov=covfun, neighborsearch=50)
kriging_neighborhood_field = Kriging.putgridpoints(xs, ys, z0)
fig, ax = PyPlot.subplots()
cax = ax.imshow(permutedims(kriging_neighborhood_field), extent=[0, 10, 0, 10], origin="lower")
for i in axes(X, 2)
	ax.plot([X[1, i]], [X[2, i]], "r.", ms=10)
end
fig.colorbar(cax)
PyPlot.title("Kriging neighborhood")
display(fig); println()
fig.savefig("kriging_neighborhood.pdf")
PyPlot.close(fig)

inversedistancefield = Array{Float64}(undef, length(xs), length(ys))
@time for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
	inversedistancefield[i, j] = Kriging.inversedistance(permutedims([x y]), X, Z)[1]
end

fig, ax = PyPlot.subplots()
cax = ax.imshow(permutedims(inversedistancefield), extent=[0, 10, 0, 10], origin="lower")
for i in axes(X, 2)
	ax.plot([X[1, i]], [X[2, i]], "r.", ms=10)
end
fig.colorbar(cax)
PyPlot.title("Inverse distance")
display(fig); println()
fig.savefig("inversedistance.pdf")
PyPlot.close(fig)

@time z0 = Kriging.interpolate_neighborhood(x0mat, X, Z; neighborsearch=50, interpolate=Kriging.inversedistance, pow=2)
inversedistance_neighborhood_field = Kriging.putgridpoints(xs, ys, z0)
fig, ax = PyPlot.subplots()
cax = ax.imshow(permutedims(inversedistance_neighborhood_field), extent=[0, 10, 0, 10], origin="lower")
for i in axes(X, 2)
	ax.plot([X[1, i]], [X[2, i]], "r.", ms=10)
end
fig.colorbar(cax)
PyPlot.title("Inverse distance neighborhood")
display(fig); println()
fig.savefig("inversedistance_neighborhood.pdf")
PyPlot.close(fig)

@time z0 = Kriging.condsim(x0mat, X, Z, covfun, 20; neighborsearch=50)
condsimfield = Kriging.putgridpoints(xs, ys, z0)
fig, ax = PyPlot.subplots()
cax = ax.imshow(permutedims(condsimfield), extent=[0, 10, 0, 10], origin="lower")
for i in axes(X, 2)
	ax.plot([X[1, i]], [X[2, i]], "r.", ms=10)
end
fig.colorbar(cax)
PyPlot.title("Conditional Gaussian simulation")
display(fig); println()
fig.savefig("condsim.pdf")
PyPlot.close(fig)