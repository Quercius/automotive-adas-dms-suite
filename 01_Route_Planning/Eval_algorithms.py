#this code actually run the simulation, calculating, on 10 different couples of start-destination places, 
#the average number of steps needed from each algorithm to find the best path 

import osmnx as ox
import random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from Dijkstra import dijkstra
from Astar import *

def get_route_nodes(G, orig, dest):
    route = []
    curr = dest
    while curr is not None:
        route.append(curr)
        if curr == orig:
            break
        curr = G.nodes[curr].get("previous")
    return route[::-1] # Ribalta la lista per averla da start a end

def calculate_route_weight(G, route):
    """Calcola il peso totale (lunghezza) di una lista di nodi."""
    weight = 0.0
    for i in range(len(route) - 1):
        # In MultiDiGraph l'arco base è identificato da 0
        weight += G.edges[(route[i], route[i+1], 0)]["weight"]
    return weight


#run
if __name__ == "__main__":
    cities = ["Aosta, Aosta, Italy", "Turin, Piedmont, Italy"]
    
    for place_name in cities:
        print("\n=========================================")
        print("Running algorithms on ", place_name)
        print(f"Downloading map data for {place_name}, please wait...")

        G = ox.graph_from_place(place_name, network_type="drive")

        for edge in G.edges:
            maxspeed = 40
            if "maxspeed" in G.edges[edge]:
                maxspeed = G.edges[edge]["maxspeed"]
                if type(maxspeed) == list:
                    speeds = [int(speed) if speed != "walk" else 1 for speed in maxspeed]
                    maxspeed = min(speeds)
                elif type(maxspeed) == str:
                    if maxspeed == "walk": 
                        maxspeed = 1
                    else:
                        maxspeed = maxspeed.strip(" mph")
                        maxspeed = int(maxspeed)
            G.edges[edge]["maxspeed"] = maxspeed
            maxspeed_ms = maxspeed/3.6 # Convert from km/h to m/s
            # Adding the "weight" attribute (time = distance / speed)
            G.edges[edge]["weight"] = G.edges[edge]["length"] / maxspeed_ms

        for edge in G.edges:
            G.edges[edge]["dijkstra_uses"] = 0
            G.edges[edge]["astar_uses"] = 0

        print("Graph loaded successfully!")
        print("Nodes: ", len(G.nodes))
        print("Edges: ", len(G.edges))

        dijkstra_iterations_num = 0.0
        manhattan_iter_num = 0.0
        euclidean_iter_num = 0.0
        haversine_iter_num = 0.0

        dijkstra_iters = []
        manhattan_iters = []
        euclidean_iters = []
        haversine_iters = []

        valid_pairs = 0

        route_m, route_e, route_h, route_d = [], [], [], []

        while valid_pairs < 10:
            start = random.choice(list(G.nodes))
            end = random.choice(list(G.nodes))

            m_steps = astar(G, start, end, "Manhattan")

            if m_steps is None:
                continue
            
            manhattan_iter_num += m_steps
            manhattan_iters.append(m_steps) 
            route_m = get_route_nodes(G, start, end)

            e_steps = astar(G, start, end, "Euclidean") 
            euclidean_iter_num += e_steps
            euclidean_iters.append(e_steps) 
            route_e = get_route_nodes(G, start, end)
            
            h_steps = astar(G, start, end, "Haversine")
            haversine_iter_num += h_steps
            haversine_iters.append(h_steps) 
            route_h = get_route_nodes(G, start, end)
            
            d_steps = dijkstra(G, start, end) 
            dijkstra_iterations_num += d_steps
            dijkstra_iters.append(d_steps)
            route_d = get_route_nodes(G, start, end)

            valid_pairs += 1

        dijkstra_average = dijkstra_iterations_num / 10
        manhattan_average = manhattan_iter_num / 10
        euclidean_average = euclidean_iter_num / 10
        haversine_average = haversine_iter_num / 10

        print(f"\n--- Average Number of Iterations ---")
        print(f"Dijkstra: {dijkstra_average}")
        print(f"A* with Manhattan heuristic: {manhattan_average}")
        print(f"A* with Euclidean heuristic: {euclidean_average}")
        print(f"A* with Haversine heuristic: {haversine_average}")

        print(f"\n--- Last Route Weights ---")
        print(f"Dijkstra: {calculate_route_weight(G, route_d):.2f}m")
        print(f"A* with Manhattan heuristic: {calculate_route_weight(G, route_m):.2f}m")
        print(f"A* with Euclidean heuristic: {calculate_route_weight(G, route_e):.2f}m")
        print(f"A* with Haversine heuristic: {calculate_route_weight(G, route_h):.2f}m")

        print(f"\nPlotting the last 4 paths found in {place_name}...")
        
        fig, ax = ox.plot_graph_routes(
            G, 
            routes=[route_d, route_m, route_e, route_h], 
            route_colors=['red', 'blue', 'green', 'yellow'], 
            route_linewidth=4, 
            node_size=0,
            bgcolor="black",
            show=False,  
            close=False
        )

        legend_elements = [
            mlines.Line2D([], [], color='red', lw=4, label='Dijkstra'),
            mlines.Line2D([], [], color='blue', lw=4, label='Manhattan'),
            mlines.Line2D([], [], color='green', lw=4, label='Euclidean'),
            mlines.Line2D([], [], color='yellow', lw=4, label='Haversine')
        ]
        ax.legend(handles=legend_elements, loc='upper right', facecolor='black', edgecolor='white', labelcolor='white', fontsize='small')

        plt.show()

    print("Done")