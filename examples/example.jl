import Kriging
import PyPlot

srand(0)
X = [3.0 7 1 5 9; 3 2 7 9 6]
Z = exp.(randn(size(X, 2)))
fig, ax = PyPlot.subplots()
for i = 1:size(X, 2)
	ax[:plot]([X[1, i]], [X[2, i]], "r.", ms=10)
	ax[:text]([X[1, i] + 0.1], [X[2, i] + 0.1], "$(Z[i])"[1:3], fontsize=16)
end
ax[:plot]([5], [5], "r.", ms=10)
ax[:text]([5.1], [5.1], "?", fontsize=16)
ax[:axis]("scaled")
ax[:set_xlim](0, 10)
ax[:set_ylim](0, 10)
display(fig); println()
fig[:savefig]("data.pdf")
PyPlot.close(fig)

covfun(h) = Kriging.expcov(h, 0.1, 3)
xs = collect(linspace(0, 10, 100))
ys = collect(linspace(0, 10, 100))
krigedfield = Array{Float64}(length(xs), length(ys))
@time for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
	krigedfield[i, j] = Kriging.krige([x y]', X, Z, covfun)[1]
end
fig, ax = PyPlot.subplots()
cax = ax[:imshow](krigedfield', extent=[0, 10, 0, 10], origin="lower")
for i = 1:size(X, 2)
	ax[:plot]([X[1, i]], [X[2, i]], "r.", ms=10)
end
fig[:colorbar](cax)
display(fig); println()
fig[:savefig]("kriging.pdf")
PyPlot.close(fig)
x0mat = Array{Float64}(2, length(xs) * length(ys))
k = 1
for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
	x0mat[1, k] = x
	x0mat[2, k] = y
	k += 1
end
@time z0 = Kriging.condsim(x0mat, X, Z, covfun, 20, length(Z); neighborsearch=50)
condsimfield = Array{Float64}(length(xs), length(ys))
k = 1
for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
	condsimfield[i, j] = z0[k]
	k += 1
end
fig, ax = PyPlot.subplots()
cax = ax[:imshow](condsimfield', extent=[0, 10, 0, 10], origin="lower")
for i = 1:size(X, 2)
	ax[:plot]([X[1, i]], [X[2, i]], "r.", ms=10)
end
fig[:colorbar](cax)
display(fig); println()
fig[:savefig]("condsim.pdf")
PyPlot.close(fig)
