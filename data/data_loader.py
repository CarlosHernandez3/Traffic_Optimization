import traci
# import Config
from pathlib import Path 

# Start SUMO with a configuration file

sumo_config_path = Path(__file__).resolve().parent.parent / "Config" / "sumo_config.sumocfg"
sumo_cmd = ["sumo", "-c", str(sumo_config_path)]

# sumo_cmd = ['sumo', '-c', 'Config/sumo_config.sumocfg']
traci.start(sumo_cmd)

try:
    step = 0
    while step < 80:  # Run simulation for 80 steps
        traci.simulationStep()  # Advance the simulation by one step

        # Get the list of all edges in the network
        edges = traci.edge.getIDList()

        # Collect traffic levels (number of vehicles) for each edge
        traffic_levels = {}
        for edge_id in edges:
            vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)  # Get the number of vehicles on the edge
            traffic_levels[edge_id] = vehicle_count

        # Print or process the traffic levels
        print(f"Step {step}: Traffic levels: {traffic_levels}")

        # Increment simulation step
        step += 1

finally:
    # Close the connection to SUMO
 traci.close()