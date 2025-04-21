import traci
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.GNN import GNN  # Your GNN model class

def build_gnn_input(tls_ids, junction_waits, junction_traffic):
    node_features = []
    for tls_id in tls_ids:
        wait = junction_waits.get(tls_id, 0.0)
        traffic = junction_traffic.get(tls_id, 0)
        avg_wait = wait / traffic if traffic > 0 else 0.0
        node_features.append([wait, traffic, avg_wait, 1.0])

    node_features = torch.tensor(node_features, dtype=torch.float, requires_grad=True)
    edge_index = []
    for i in range(len(tls_ids)):
        for j in range(len(tls_ids)):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return node_features, edge_index

# SUMO configuration
sumo_config_path = Path(__file__).resolve().parent.parent / "Config" / "sumo_config.sumocfg"
sumo_cmd = ["sumo-gui", "-c", str(sumo_config_path)]
traci.start(sumo_cmd)

# GNN setup
input_dim = 4
hidden_dim = 8
output_dim_duration = 1
output_dim_phase = 3
model = GNN(input_dim, hidden_dim, output_dim_duration, output_dim_phase)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

try:
    for step in range(200):
        traci.simulationStep()

        tls_ids = traci.trafficlight.getIDList()
        junction_waiting_time = {tls_id: 0.0 for tls_id in tls_ids}
        traffic_levels = {tls_id: 0 for tls_id in tls_ids}

        vehicle_ids = traci.vehicle.getIDList()
        for veh_id in vehicle_ids:
            wait = traci.vehicle.getWaitingTime(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)
            controlled_tls = traci.trafficlight.getIDList()
            for tls in controlled_tls:
                controlled_lanes = traci.trafficlight.getControlledLanes(tls)
                if lane_id in controlled_lanes:
                    junction_waiting_time[tls] += wait
                    traffic_levels[tls] += 1
                    break

        node_features, edge_index = build_gnn_input(tls_ids, junction_waiting_time, traffic_levels)

        if node_features.size(0) > 0 and edge_index.size(1) > 0:
            duration_output, phase_output_raw = model(node_features, edge_index)
            duration_output = torch.relu(duration_output).squeeze()
            phase_output = torch.argmax(phase_output_raw, dim=1)

            for i, tls_id in enumerate(tls_ids):
                max_phase = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases) - 1
                phase = min(max_phase, int(phase_output[i].item()))
                duration = max(5.0, min(60.0, float(duration_output[i].item())))

                traci.trafficlight.setPhase(tls_id, phase)
                traci.trafficlight.setPhaseDuration(tls_id, duration)

            optimizer.zero_grad()
            loss = torch.tensor(0.0, requires_grad=True)
            for tls_id in tls_ids:
                wait = junction_waiting_time[tls_id]
                traffic = traffic_levels.get(tls_id, 1.0)
                loss = loss + wait * torch.log1p(torch.tensor(traffic, dtype=torch.float))

            loss.backward()
            optimizer.step()

            print(f"Step {step}: Loss: {loss.item():.4f}")
            print(f"Phase Output: {phase_output.tolist()}")
            print(f"Duration Output: {[round(float(x), 2) for x in duration_output.tolist()]}")

        else:
            print(f"Step {step}: Skipping GNN forward pass (no data)")

finally:
    traci.close()
