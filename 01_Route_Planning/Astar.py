#This is a modified version of the Dijkstra script to perform the A* algorithm with the 3 different euristics

import osmnx as ox
import random
import heapq
import math



### style

def style_unvisited_edge(G, edge):        
    G.edges[edge]["color"] = "gray"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 0.2

def style_visited_edge(G, edge):
    G.edges[edge]["color"] = "green"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(G, edge):
    G.edges[edge]["color"] = "red"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_path_edge(G, edge, color="white"): 
    G.edges[edge]["color"] = color
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 5

def plot_graph(G):
    ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "black"
    )



# distance heuristics (h)

def manhattan_distance(x1, y1, x2, y2):
    lat_m = abs(y2 - y1) * 111320
    lon_m = abs(x2 - x1) * 111320 
    return lat_m + lon_m


def euclidean_distance(x1, y1, x2, y2):
    lat_m = (y2 - y1) * 111320
    lon_m = (x2 - x1) * 111320 
    return math.sqrt(lat_m**2 + lon_m**2)


def haversine_distance(x1, y1, x2, y2):
    phi1 = math.radians(y1)
    phi2 = math.radians(y2)
    d_phi = math.radians(y2 - y1)
    d_lambda = math.radians(x2 - x1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * 6371000 * math.atan2(math.sqrt(a), math.sqrt(1 - a))



# heuristic calculation method

def calculate_h(G, node_current, node_target, heuristic_name):
    x1, y1 = G.nodes[node_current]['x'], G.nodes[node_current]['y']
    x2, y2 = G.nodes[node_target]['x'], G.nodes[node_target]['y']
    
    match heuristic_name:
        case "Manhattan":
            dist_m = manhattan_distance(x1, y1, x2, y2)
        case "Euclidean":
            dist_m = euclidean_distance(x1, y1, x2, y2)
        case "Haversine":
            dist_m = haversine_distance(x1, y1, x2, y2)

    MAX_SPEED_MS = 40 / 3.6 
    return dist_m / MAX_SPEED_MS


# A* algorithm

def astar(G, orig, dest, heuristic, plot=False):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(G, edge)
    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50

    h_start = calculate_h(G, orig, dest, heuristic) 
    pq = [(h_start, orig)] #just the heuristic for the fist node

    step = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            #print("Iterations:", step)
            #plot_graph()
            return step 
        if G.nodes[node]["visited"]: continue
        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node):
            style_visited_edge(G, (edge[0], edge[1], 0))
            neighbor = edge[1]
            weight = G.edges[(edge[0], edge[1], 0)]["weight"]
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node

                h_neighbor = calculate_h(G, neighbor, dest, heuristic)
                f_score = G.nodes[neighbor]["distance"] + h_neighbor
                heapq.heappush(pq, (f_score, neighbor))

                for edge2 in G.out_edges(neighbor):
                    style_active_edge(G, (edge2[0], edge2[1], 0))
        step += 1



# path reconstruction

def reconstruct_path(orig, dest, plot=False, algorithm=None):
    for edge in G.edges:
        style_unvisited_edge(G, edge)
    dist = 0
    speeds = []
    curr = dest
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        dist += G.edges[(prev, curr, 0)]["length"]
        speeds.append(G.edges[(prev, curr, 0)]["maxspeed"])
        style_path_edge(G, (prev, curr, 0))
        if algorithm:
            G.edges[(prev, curr, 0)][f"{algorithm}_uses"] = G.edges[(prev, curr, 0)].get(f"{algorithm}_uses", 0) + 1
        curr = prev
    dist /= 1000



### run
if __name__ == "__main__":
    cities = ["Aosta, Aosta, Italy", "Turin, Piedmont, Italy"]
    distance_heuristics = ["Manhattan", "Euclidean", "Haversine"]

    for place_name in cities:

        G = ox.graph_from_place(place_name, network_type="drive")

        for edge in G.edges:
            # Cleaning the "maxspeed" attribute, some values are lists, some are strings, some are None
            maxspeed = 40
            if "maxspeed" in G.edges[edge]:
                maxspeed = G.edges[edge]["maxspeed"]
                if type(maxspeed) == list:
                    #speeds = [ int(speed) for speed in maxspeed ]
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
            G.edges[edge]["astar_uses"] = 0

        print("Running A* on ", place_name);
        print("Nodes: ", len(G.nodes))
        print("Edges: ", len(G.edges))

        manhattan_iter_num = 0.0
        euclidean_iter_num = 0.0
        haversine_iter_num = 0.0

        valid_pairs = 0

        while valid_pairs < 10:
            start = random.choice(list(G.nodes))
            end = random.choice(list(G.nodes))

            m_steps = astar(G, start, end, "Manhattan") #controllo solo su manhattan se il percorso è valido...

            if m_steps is None:
                continue
            
            manhattan_iter_num += m_steps

            #se lo è faccio anche per le altre due euristiche
            euclidean_iter_num += astar(G, start, end, "Euclidean")
            haversine_iter_num += astar(G, start, end, "Haversine")

            valid_pairs += 1
                
        manhattan_average = manhattan_iter_num/10
        euclidean_average = euclidean_iter_num/10
        haversine_average = haversine_iter_num/10

        for heuristic in distance_heuristics:
            print("Iterations average using ", heuristic, " distance is", locals().get(f"{heuristic.lower()}_average"))
        

        paths_edges = []
        # 1. Ricalcolo e salvo in memoria gli archi dei 3 percorsi
        for heuristic in distance_heuristics:
            astar(G, start, end, heuristic)
            path = []
            total_weight = 0.0  # <-- AGGIUNTA: variabile per accumulare il peso del percorso
            
            curr = end
            while curr != start:
                prev = G.nodes[curr]["previous"]
                edge = (prev, curr, 0)
                path.append(edge)
                total_weight += G.edges[edge]["weight"]  # <-- AGGIUNTA: somma il peso dell'arco
                curr = prev
                
            paths_edges.append(path)
            print("Peso totale percorso", heuristic, ":", total_weight)  # <-- AGGIUNTA: stampa in console
            
        # 2. Ripulisco la mappa da tutto il verde/rosso dell'ultima esecuzione
        for edge in G.edges:
            style_unvisited_edge(G, edge)
            
        # 3. Disegno i 3 percorsi uno sopra l'altro con i 3 colori
        colors = ["red", "cyan", "yellow"]
        for path, color in zip(paths_edges, colors):
            for edge in path:
                style_path_edge(G, edge, color)
        
        plot_graph(G)

    print("Done")

