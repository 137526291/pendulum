import rebound

sim = rebound.Simulation()
sim.add(m=1.0) # Saturn
sim.add(m=1e-4, a=1.0) # Moon

for i in range(100):
    sim.add(a=0.5+0.001*i, e=0.0)

sim.integrate(100) #100 t units
rebound.OrbitPlot(sim)