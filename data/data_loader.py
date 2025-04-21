import traci
from pathlib import Path 
import torch
# from src.GNN import GNN

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.GNN import GNN

# from utils import build_gnn_input
def build_gnn_input(junction_ids, junction_waits, junction_traffic):
   
   node_features = []
   for j_id in junction_ids:
        wait = junction_waits.get(j_id, 0.0)
        traffic = junction_traffic.get(j_id, 0)
        avg_wait = wait/traffic if traffic > 0 else 0.0

        node_features.append([wait, traffic, avg_wait, 1.0])

   node_features = torch.tensor(node_features, dtype=torch.float)
   edge_index = []
   for i in range(len(junction_ids)):
        for j in range(len(junction_ids)):
            if i != j:
                edge_index.append([i, j])
   edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

   return node_features, edge_index


sumo_config_path = Path(__file__).resolve().parent.parent / "Config" / "sumo_config.sumocfg"
sumo_cmd = ["sumo-gui", "-c", str(sumo_config_path)]
traci.start(sumo_cmd)

input_dim = 4
hidden_dim = 8
output_dim_duration = 1
output_dim_phase = 3
model = GNN(input_dim, hidden_dim, output_dim_duration,output_dim_phase)

try:
    step = 0
    junction_waiting_time = {}

    while step < 200:  # Run simulation for 200 steps
        traci.simulationStep() 
        vehicle_ids = traci.vehicle.getIDList()
        print(f"{vehicle_ids}")

         # Advance the simulation by one step

        # Get the list of all edges in the network
        edges = traci.edge.getIDList()

        # Collect traffic levels (number of vehicles) for each edge
        traffic_levels = {}
        for edge_id in edges:
            vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)  # Get the number of vehicles on the edge
            traffic_levels[edge_id] = vehicle_count

        for veh_id in vehicle_ids:
           waiting_time = traci.vehicle.getWaitingTime(veh_id)
           lane_id = traci.vehicle.getLaneID(veh_id)

           junction_id = traci.lane.getEdgeID(lane_id)

           if junction_id not in junction_waiting_time:
              junction_waiting_time[junction_id] = 0

           junction_waiting_time[junction_id] += waiting_time


        junction_ids = list(junction_waiting_time.keys())        
        node_features, edge_index = build_gnn_input(junction_ids, junction_waiting_time, traffic_levels)
        print(junction_ids)
        print(f"Junction (lane-based) waiting time: {junction_waiting_time}")


        # Print or process the traffic levels
        # print(f"Step {step}: Traffic levels: {traffic_levels}"
        
        if node_features.size(0) > 0 and edge_index.size(1) > 0:
            duration_output, phase_output = model(node_features, edge_index)
            gnn_output = (duration_output, phase_output)
            tls_ids = traci.trafficlight.getIDList()

            for i, jid in enumerate(junction_ids):
                if jid in tls_ids:

                    duration = float(duration_output[i].item())
                    duration = max(5.0, min(duration, 120.0))

                    traci.trafficlight.setPhase(jid, phase_output)
                    traci.trafficlight.setPhaseDuration(jid, duration_output)
            print(f"Step {step}: GNN output:\n{gnn_output}")
            print(f"Step {step}: Waiting Times: {junction_waiting_time}")
        else:
            print(f" Skipping GNN forward pass (no data)")
        # Increment simulation step
        step += 1

finally:
    # Close the connection to SUMO
 traci.close()