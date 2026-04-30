using WaterLily, StaticArrays, Interpolations, Plots

fit = y-> scale(interpolate(y, BSpline(Quadratic(Line(OnGrid())))), range(0,1,length=length(y)))


	
nose,len = (30,224),500
width = [0.02,0.07,0.06,0.048,0.03,0.019,0.01]
scatter!(nose[1].+len.*range(0,1,length=length(width)),
    nose[2].-len.*width,color=:blue,legend=false)
thk = fit(width)
x = 0:0.01:1
plot(nose[1].+len.*x, [nose[2].-len.*thk.(x),nose[2].+len.*thk.(x)],color=:blue)


envelope = [0.2,0.21,0.23,0.4,0.88,1.0]
amp = fit(envelope)

λ = 0.2

scatter(0:0.2:1, envelope)
colors = palette(:cyclic_wrwbw_40_90_c42_n256)
for t in 1/12:1/12:1
    plot!(x,amp.(x).*sin.(2π/λ*x.-2π*t),color=colors[floor(Int,t*256)])
end
plot!(ylim=(-1.4,1.4),legend=false)

shift = 1


T = 0.8

function segment_sdf(x,y)
    s = clamp(x,0,1)         # distance along the segment
    y = y-shift              # shift laterally
    sdf = √sum(abs2,(x-s,y)) # line segment SDF
    return sdf-T*thk(s)      # subtract thickness
end
grid = -1:0.05:2
contourf(grid,grid,segment_sdf,clim=(-1,2),linewidth=0)
contour!(grid,grid,segment_sdf,levels=[0],color=:black) # zero contour


function fish(thk,amp,k=5.3;L=2^6,A=0.1,St=0.3,Re=1e4)
    # fraction along fish length
    s(x) = clamp(x[1]/L,0,1)

    # fish geometry: thickened line SDF
    sdf(x,t) = √sum(abs2,x-L*SVector(s(x),0.))-L*thk(s(x))

    # fish motion: travelling wave
    U=1
    ω = 2π*St*U/(2A*L)
    function map(x,t)
        xc = x.-L # shift origin
        return xc-SVector(0.,A*L*amp(s(xc))*sin(k*s(xc)-ω*t))
    end

    # make the fish simulation
    return Simulation((256, 128),(U,0.),L;
                        ν=U*L/Re,body=AutoBody(sdf,map))
end

# Create the swimming shark
L,A,St = 3*2^5,0.1,0.3
swimmer = fish(thk,amp;L,A,St);

# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0,23/24*period,length=24)

@gif for t ∈ cycle
	measure!(swimmer,t*swimmer.L/swimmer.U);
	contour(swimmer.flow.μ₀[:,:,1]',
		aspect_ratio=:equal,legend=false,border=:none)
end

sim_step!(swimmer,0.1,remeasure=true)
sim_time(swimmer)

function plot_vorticity(sim)
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    contourf(sim.flow.σ',
        color=palette(:BuGn), clims=(-10,10),linewidth=0,
        aspect_ratio=:equal,legend=false,border=:none)
end

# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer).+cycle
    sim_step!(swimmer,t,remeasure=true)
    plot_vorticity(swimmer)
end

function get_force(sim, t)
    sim_step!(sim, t, remeasure=true)
    return WaterLily.total_force(sim)
end



force = WaterLily.total_force(swimmer)
forces = [get_force(swimmer,t) for t ∈ sim_time(swimmer).+cycle]
"got forces"

# scatter(cycle ./ period, [first.(forces), last.(forces)],
#     labels=permutedims(["thrust", "side"]),
#     xlabel="scaled time", ylabel="scaled force")

scatter(cycle./period,[first.(forces),last.(forces)],
	labels=permutedims(["thrust","side"]),
	xlabel="scaled time", ylabel="scaled force")
