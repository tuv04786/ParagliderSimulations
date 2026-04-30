using WaterLily, StaticArrays, GLMakie, CUDA

# Parameters
AoA = 0.0  # Angle of attack in degrees
Re = 4e4   # Reynolds number
U = 1.0    # Freestream velocity
c = 1.0    # Chord length
ν = U * c / Re  # Kinematic viscosity

# Domain setup
n = 512  # Grid resolution
m = div(n, 4)  # Number of multi-grid levels
L = 8 * c  # Domain size

# Create NACA0022 airfoil
function naca_airfoil(t, n_points=200)
    m = 0.02  # max camber (0%)
    p = 0.4   # position of max camber
    
    x = range(0, 1, length=n_points)
    y_c = zeros(n_points)
    dy_c = zeros(n_points)
    
    for i in eachindex(x)
        if x[i] < p
            y_c[i] = m / p^2 * (2 * p * x[i] - x[i]^2)
            dy_c[i] = 2 * m / p^2 * (p - x[i])
        else
            y_c[i] = m / (1 - p)^2 * ((1 - 2 * p) + 2 * p * x[i] - x[i]^2)
            dy_c[i] = 2 * m / (1 - p)^2 * (p - x[i])
        end
    end
    
    θ = atan.(dy_c)
    y_t = t / 0.2 * (0.2969 * sqrt.(x) .- 0.1260 * x .- 0.3516 * x.^2 .+ 0.2843 * x.^3 .- 0.1015 * x.^4)
    
    x_u = x .- y_t .* sin.(θ)
    y_u = y_c .+ y_t .* cos.(θ)
    x_l = x .+ y_t .* sin.(θ)
    y_l = y_c .- y_t .* cos.(θ)
    
    return vcat(reverse(x_u), x_l), vcat(reverse(y_u), y_l)
end

x_foil, y_foil = naca_airfoil(0.022)
x_foil .= x_foil .* c .- 0.25 * c
y_foil .= y_foil .* c

# Create simulation
function create_body(n, x_foil, y_foil, AoA_rad)
    angle = AoA_rad
    x_rot = x_foil .* cos(angle) - y_foil .* sin(angle)
    y_rot = x_foil .* sin(angle) + y_foil .* cos(angle)

    points = map(i -> SVector(x_rot[i], y_rot[i]), eachindex(x_foil))
    return CustomBody(points)
end

body = create_body(n, x_foil, y_foil, deg2rad(AoA))

# Initialize simulation on GPU
sim = Simulation((n, n), (U, 0), ν; body=body, mem=CUDA)

# Time stepping
n_steps = 5000
sample_freq = 10

U_history = []
p_history = []
ω_history = []

@time for i = 1:n_steps
    sim_step!(sim)
    
    if i % sample_freq == 0
        push!(U_history, Array(sim.flow.u))
        push!(p_history, Array(sim.flow.p))
        push!(ω_history, Array(sim.flow.ω))
    end
    
    i % 500 == 0 && println("Step $i/$(n_steps)")
end

# Extract forces
cd, cl = WaterLily.drag(sim), WaterLily.lift(sim)

println("\nResults:")
println("Cd = $cd")
println("Cl = $cl")

# Create visualization
fig = Figure(size=(1200, 800))

# Velocity field
ax1 = Axis(fig[1, 1], title="Velocity Magnitude")
U_final = sqrt.(Array(sim.flow.u[1, :, :]).^2 .+ Array(sim.flow.u[2, :, :]).^2)
hm1 = heatmap!(ax1, U_final, colormap=:viridis)
Colorbar(fig[1, 2], hm1)

# Pressure field
ax2 = Axis(fig[2, 1], title="Pressure Distribution")
p_final = Array(sim.flow.p)
hm2 = heatmap!(ax2, p_final, colormap=:coolwarm)
Colorbar(fig[2, 2], hm2)

# Vorticity field
ax3 = Axis(fig[1, 3], title="Vorticity")
ω_final = Array(sim.flow.ω)
hm3 = heatmap!(ax3, ω_final, colormap=:RdBu)
Colorbar(fig[1, 4], hm3)

# Forces text
ax4 = Axis(fig[2, 3], title="Forces")
text!(ax4, 0.5, 0.7, text="Cd = $(round(cd, digits=4))", fontsize=20, align=(:center, :center))
text!(ax4, 0.5, 0.4, text="Cl = $(round(cl, digits=4))", fontsize=20, align=(:center, :center))
hidedecorations!(ax4)

save("airfoil_naca0022_results.png", fig)
display(fig)

println("Simulation complete!")