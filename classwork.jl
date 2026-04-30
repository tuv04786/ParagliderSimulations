using LinearAlgebra
using Plots

L = 1
N = 11
x = LinRange(0, L, N)

M = 21
t = LinRange(0, 1, M)


k = 1/100
dt = t[2] - t[1]
dx = x[2] - x[1]

IC(x) = sin(2pi * x / L)
plot(x, IC)

u = zeros(N, M)
u[:, 1] = IC.(x)
for i = 2:M
    u(:,1)
end