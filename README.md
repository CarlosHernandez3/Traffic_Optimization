# README: Traffic Optimization Using GNN and SUMO

## **Project Overview**
This project demonstrates how to optimize traffic light configurations in a simulated urban traffic environment using a Graph Neural Network (GNN). The simulation is powered by SUMO (Simulation of Urban Mobility), while the GNN predicts optimal traffic light timings based on congestion data.

---

## **Key Components**
1. **SUMO**: A microscopic traffic simulator used to simulate urban traffic environments.
2. **GNN**: A Graph Neural Network that models the traffic network as a graph, where nodes represent intersections, and edges represent roads.
3. **XML Input Files**: Custom traffic network configurations and scenarios defined in XML format for SUMO.

---

## **Steps to Run the Project**

### **1. Install Dependencies**
- Install SUMO: Follow the [SUMO installation guide](https://www.eclipse.org/sumo/) to set up SUMO on your machine.
- Install Python libraries:
  ```bash
  pip install torch 
  ```

### **2. Create Traffic Network and Simulation Data**
- Define the traffic network using SUMO’s XML format. Create files such as:
  - **Net file**: Defines the road network.
  - **Routes file**: Specifies vehicle routes.
  - **Config file**: Links the above files for the simulation.
  - **Edges file**: Specifies roads in network
  - **Node file**: Specifies intersections in road network
- Example files:
  - `network.net.xml`
  - `routes.rou.xml`
  - `config.sumocfg`
  - `edges.xml`
  - `nodes.xml`

### **3. Generate Simulated Traffic Data**
1. Run SUMO to simulate traffic scenarios:
   ```bash
   sumo-gui -n ./Config/network.net.xml
   sumo -c config.sumocfg
   ```
2. Extract traffic metrics such as vehicle counts, queue lengths, and speeds using the TraCI Python API:
   ```python
   import traci
   traci.start(['sumo', '-c', 'config.sumocfg'])
   while traci.simulation.getMinExpectedNumber() > 0:
       traci.simulationStep()
       vehicle_count = traci.edge.getLastStepVehicleNumber('edge_id')
   traci.close()
   ```

### **4. Prepare the Traffic Graph**
- Represent intersections as nodes and roads as edges.
- Use SUMO’s outputs to calculate features for nodes and edges, such as:
  - **Node features**: Current traffic light state, queue length.
  - **Edge features**: Vehicle count, average speed.

### **5. Train the GNN**
- Use the provided GNN architecture (`GVAE`) to train a model on simulated data:
  - **Input**: Graph representation of traffic.
  - **Output**: Predicted traffic light configurations.
- Train the model:
  ```python
  
  ```

### **6. Evaluate the GNN**
- Test the trained GNN on unseen traffic scenarios.
- Use SUMO to simulate the predicted configurations and evaluate performance metrics such as:
  - Average waiting time.
  - Vehicle throughput.

### **7. Deployment**
- Integrate the GNN predictions into SUMO using TraCI for real-time traffic light control.
  ```python
  ```

---

## **Project Structure**
```
|-- src/
    |-- gnn.py       # GNN implementation
    |-- train.py       # Training script
    |-- xmlConverter.py   # SUMO simulation scripts
|-- data/
    |-- network.net.xml    # Traffic network file
    |-- routes.rou.xml     # Routes file
    |-- config.sumocfg     # SUMO config file
    |-- edges.xml          # Edges config file
    |-- nodes.xml          # Nodes config file

|-- results/
    |-- logs/              # Training logs
    |-- models/            # Saved GNN models
```


---

## **References**
- [SUMO Documentation](https://sumo.dlr.de/docs/index.html)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/)
- [xml.etree Documentation](https://docs.python.org/3/library/xml.etree.elementtree.html)

