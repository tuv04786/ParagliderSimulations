using WaterLily
using StaticArrays
using LinearAlgebra
using CUDA
using WriteVTK

# ==========================================
# 1. Geometry Math (GPU/Float32 Safe)
# ==========================================

function naca0022(x)
    t = 0.22
    return 5.0 * t * (0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2 + 0.2843*x^3 - 0.1015*x^4)
end

function sdSegment(p, a, b)
    pa = p - a
    ba = b - a
    T = eltype(p) 
    h = clamp(dot(pa, ba) / dot(ba, ba), T(0.0), T(1.0))
    return norm(pa - ba * h)
end

function sdSkin(p, points)
    d = eltype(p)(Inf) 
    for i in 1:(length(points)-1)
        d = min(d, sdSegment(p, points[i], points[i+1]))
    end
    return d
end

# ==========================================
# 2. Body Generation 
# ==========================================

function make_ram_air_body(L; thickness=1.5, cut_fraction=0.12, T=Float32)
    theta_top = range(0, pi, length=40)
    x_top = 0.5 .* (1.0 .- cos.(theta_top))
    top_pts = Tuple(SVector{2,T}(T(x * L), T(naca0022(x) * L)) for x in x_top)

    theta_start = acos(1.0 - 2.0 * cut_fraction)
    theta_bot = range(theta_start, pi, length=30)
    x_bot = 0.5 .* (1.0 .- cos.(theta_bot))
    bot_pts = Tuple(SVector{2,T}(T(x * L), T(-naca0022(x) * L)) for x in x_bot)

    thick_T = T(thickness)
    sdf = (p, t) -> min(sdSkin(p, top_pts), sdSkin(p, bot_pts)) - thick_T
    return sdf
end

# ==========================================
# 3. Setup Simulation
# ==========================================

function setup_simulation(; L=96, Re=2000, AoA=0.0, mem=Array, T=Float32)
    width, height = 4 * L, 3 * L 
    center = SVector{2,T}(T(1.5 * L), T(1.5 * L)) 
    sdf = make_ram_air_body(L, T=T)
    
    θ = T(-AoA * π / 180.0) 
    R = SMatrix{2,2,T}(cos(θ), -sin(θ), sin(θ), cos(θ))
    
    function map(x, t)
        return R * (x - center)
    end
    
    body = AutoBody(sdf, map)
    return Simulation((width, height), (T(1.0), T(0.0)), L; ν=T(1.0/Re), body=body, T=T, mem=mem)
end

# ==========================================
# 3.5 Math Helpers
# ==========================================

function calc_vorticity(u_cpu, N, M)
    vort = zeros(Float32, N, M)
    for x in 2:N-1, y in 2:M-1
        vort[x, y] = (u_cpu[x+1, y, 2] - u_cpu[x-1, y, 2]) - (u_cpu[x, y+1, 1] - u_cpu[x, y-1, 1])
    end
    return vort
end

# ==========================================
# 4. Fast GPU Export to ParaView
# ==========================================

L_chord = 128  
t_step = 0.1 
n_frames = 100 

println("Initializing GPU Simulation...")

sim = setup_simulation(L=L_chord, Re=1*10^6, AoA=9.0, mem=CuArray, T=Float32)
N_grid, M_grid = size(sim.flow.p)

# We will save all frames into a dedicated folder to keep things clean
output_dir = "paraview_output"
mkpath(output_dir)

println("Running physics and exporting VTK data...")

for i in 1:n_frames
    # 1. RUN PHYSICS ON THE GPU
    sim_step!(sim, Float32(i * t_step))
    
    # 2. PULL DATA TO CPU
    u_cpu = Array(sim.flow.u)
    p_cpu = Array(sim.flow.p)
    
    # Anchor the far-field pressure to exactly 0.0 for stable ParaView colors
    p_cpu .-= p_cpu[2, 2]
    
    # Calculate the vorticity on the CPU
    vort_cpu = calc_vorticity(u_cpu, N_grid, M_grid)
    
    # 3. DUMP TO VTK 
    filename = joinpath(output_dir, "frame_$(lpad(i, 4, '0'))")
    
    vtk_grid(filename, 1:N_grid, 1:M_grid) do vtk
        # Give ParaView the Pressure scalar
        vtk["Pressure"] = p_cpu
        
        # Give ParaView the Velocity vector 
        vtk["Velocity"] = (u_cpu[:, :, 1], u_cpu[:, :, 2])
        
        # Give ParaView the calculated Vorticity!
        vtk["Vorticity"] = vort_cpu
        
        # Give ParaView the Body Distance map so you can draw the outline
        vtk["Body_SDF"] = [WaterLily.sdf(sim.body, SVector{2,Float32}(Float32(x), Float32(y)), 0f0) for x in 1:N_grid, y in 1:M_grid]
    end
    
    if i % 10 == 0
        println("Exported Frame $i of $n_frames")
    end
end

println("Done! Data is ready for ParaView.")