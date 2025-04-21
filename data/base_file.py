import traci
from pathlib import Path

# Build GNN input removed — not needed for basic SUMO run

sumo_config_path = Path(__file__).resolve().parent.parent / "Config" / "sumo_config.sumocfg"
sumo_cmd = ["sumo-gui", "-c", str(sumo_config_path)]
traci.start(sumo_cmd)

try:
    step = 0
    junction_waiting_time = {}
    vehicle_trip_times = {}
    completed_trip_times = [] 

    while step < 800:
        traci.simulationStep()
        vehicle_ids = traci.vehicle.getIDList()
        print(f"Vehicles at step {step}: {vehicle_ids}")

        # Get the list of all edges in the network
        edges = traci.edge.getIDList()

        traffic_levels = {}
        for edge_id in edges:
            vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
            traffic_levels[edge_id] = vehicle_count



        departed_vehicles = traci.simulation.getDepartedIDList()
        for veh_id in departed_vehicles:
            vehicle_trip_times[veh_id] = step 

        arrived_vehicles = traci.simulation.getArrivedIDList()
        for veh_id in arrived_vehicles:
            if veh_id in vehicle_trip_times:
                trip_time = step - vehicle_trip_times[veh_id]
                completed_trip_times.append(trip_time)



        for veh_id in vehicle_ids:
            waiting_time = traci.vehicle.getWaitingTime(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)
            junction_id = traci.lane.getEdgeID(lane_id)

            if junction_id not in junction_waiting_time:
                junction_waiting_time[junction_id] = 0

            junction_waiting_time[junction_id] += waiting_time

        if completed_trip_times:
            avg_sum_time = sum(completed_trip_times)/len(completed_trip_times)
            print(f"Average trip time: {avg_sum_time:.2f} seconds")

        step += 1

finally:
    traci.close()