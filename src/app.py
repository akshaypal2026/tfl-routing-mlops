import streamlit as st
import networkx as nx
import json
import os
import sys

# Add src to path so we can import data_harvest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.data_harvest import harvest_real_tfl_data
except ImportError:
    pass

# Streamlit App Configuration
st.set_page_config(page_title="TfL Comfort-Adjusted Routing", layout="wide")
st.title("🚇 TfL Passenger Density Forecasting & Routing")
st.markdown("A serverless MLOps pipeline using GNNs to provide comfort-adjusted paths.")

# Load Data
@st.cache_data
def load_graph_data():
    data_path = "data/raw/tfl_density_latest.json"
    if not os.path.exists(data_path):
        st.info("Data not found. Harvesting latest TfL data on the fly...")
        try:
            harvest_real_tfl_data()
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return None
    
    with open(data_path, "r") as f:
        data = json.load(f)
        
    G = nx.Graph()
    # Assuming baseline travel time is 5 minutes for all edges for simplicity
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"], 
                   travel_time=5, 
                   forecast_density=edge["density"])
    return G, data

data_tuple = load_graph_data()

if data_tuple:
    G, raw_data = data_tuple
    
    st.sidebar.header("Routing Settings")
    nodes = list(G.nodes())
    start_node = st.sidebar.selectbox("Start Station", nodes, index=0)
    end_node = st.sidebar.selectbox("End Station", nodes, index=len(nodes)-1)
    
    alpha = st.sidebar.slider("Comfort Sensitivity (α)", 0.0, 5.0, 1.0, 0.1,
                              help="0 = Fastest route (ignore crowding). Higher = Avoid crowds.")

    # Custom A* Weight Function
    def comfort_adjusted_weight(u, v, d):
        # Cost = TravelTime * (1 + alpha * DensityForecast)
        travel_time = d.get('travel_time', 5)
        density = d.get('forecast_density', 0.5)
        return travel_time * (1 + alpha * density)

    # Heuristic for A* (Euclidean distance could be used here, but for our simple graph, 0 is fine, making it Dijkstra)
    def heuristic(u, v):
        return 0

    if st.sidebar.button("Find Route"):
        if start_node == end_node:
            st.warning("Start and end stations are the same.")
        else:
            try:
                # Fastest Path (alpha = 0)
                fastest_path = nx.astar_path(G, start_node, end_node, 
                                            heuristic=heuristic, 
                                            weight=lambda u,v,d: d.get('travel_time', 5))
                fastest_cost = sum([G[u][v]['travel_time'] for u, v in zip(fastest_path[:-1], fastest_path[1:])])
                
                # Comfort Adjusted Path
                comfort_path = nx.astar_path(G, start_node, end_node, 
                                            heuristic=heuristic, 
                                            weight=comfort_adjusted_weight)
                
                comfort_time = sum([G[u][v]['travel_time'] for u, v in zip(comfort_path[:-1], comfort_path[1:])])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success("Fastest Path")
                    st.write(" $\\rightarrow$ ".join(fastest_path))
                    st.metric("Estimated Travel Time", f"{fastest_cost} mins")
                    
                with col2:
                    st.info(f"Comfort-Adjusted Path (α={alpha})")
                    st.write(" $\\rightarrow$ ".join(comfort_path))
                    st.metric("Estimated Travel Time", f"{comfort_time} mins", 
                              delta=f"+{comfort_time - fastest_cost} mins (trade-off)", delta_color="inverse")
                    
                # Visualize Graph
                st.subheader("Network Density Visualization")
                # Create a simple representation using graphviz/streamlit
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Colors based on density
                edge_colors = [G[u][v]['forecast_density'] for u,v in G.edges()]
                pos = nx.spring_layout(G, seed=42)
                
                nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, 
                        font_weight='bold', edge_color=edge_colors, width=3, edge_cmap=plt.cm.RdYlGn_r, ax=ax)
                
                # Highlight path
                path_edges = list(zip(comfort_path, comfort_path[1:]))
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=5, ax=ax)
                
                st.pyplot(fig)
                
            except nx.NetworkXNoPath:
                st.error("No path found between these stations.")
