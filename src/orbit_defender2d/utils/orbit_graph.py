# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

''' encoding board as a graph in order to capture relationships between board locations
    this data structures will be used to determine piece adjacencies, compute temporally extended actions, 
    look up values that determine action outcomes '''


import networkx as nx
import numpy as np
import itertools
from math import floor

import orbit_defender2d.utils.utils as U

class OrbitGraph:


    def __init__(self, orbitGrid, FuelUsage=None):

        ''' Instantiate graph with sector as nodes and moves/attacks/defense as edges

        Args:
            engagements (dict): key is piece id, one for each piece in game
                                value is the piece's engagement tuple (engagement_type, target_piece_id, probability)
        '''

        # graph capturing grid adjacency along edges of sectors (i.e., corner-only adjacency not counted)
        self.adj_graph = nx.Graph()
        self.adj_graph.add_nodes_from(range(orbitGrid.n_sectors))

        # graph capturing movement transitions available from each node in game's 'movement phase' and cost of move in terms of fuel
        self.move_graph = nx.DiGraph()
        self.move_graph.add_nodes_from(range(orbitGrid.n_sectors))

        # graph capturing transitions available from each node though movement and drift phase, and associated cost of move in fuel
        self.turn_graph = nx.DiGraph()
        self.turn_graph.add_nodes_from(range(orbitGrid.n_sectors))

        # if FuelUsage not passed in, create a default based on koth
        if FuelUsage is None:           
            FuelUsage = {
                            U.NOOP: 0.0,
                            U.DRIFT: 1.0,
                            U.PROGRADE: 5.0,
                            U.RETROGRADE: 10.0,
                            U.RADIAL_IN: 1.0,
                            U.RADIAL_OUT: 1.0,
                            U.IN_SEC:{
                                U.SHOOT: 5.0,
                                U.COLLIDE: 20.0,
                                U.GUARD: 20.0
                            },
                            U.ADJ_SEC:{
                                U.SHOOT: 5.0,
                                U.COLLIDE: 30.0,
                                U.GUARD: 30.0
                            }
                        }
        
        if U.DRIFT not in FuelUsage:      #if drift not present add it
            FuelUsage[U.DRIFT]=1.0

        for nodes in itertools.product(list(self.move_graph),list(self.move_graph)):
                
                (ring0, azim0) = orbitGrid.sector_num2coord(sec_num=nodes[0])
                (ring1, azim1) = orbitGrid.sector_num2coord(sec_num=nodes[1])
                
                # adjacency graph
                if ring0 == ring1:                                  #if in same ring
                    if abs((azim0-azim1)%(2**ring0)) <= 1:                 #if within distance 1 of azimuth
                        self.adj_graph.add_edge(nodes[0],nodes[1])     #add edge
                if abs(ring0-ring1) == 1:
                    if floor(azim0/2)+1 == azim1 or 2*azim0+1 == azim1 or 2*azim0 == azim1:
                        self.adj_graph.add_edge(nodes[0],nodes[1])
                
                # movement graph, assume object at node[0] moving to node[1]
                if ring0 == ring1:                                                                                 #if in same ring
                    if (azim0-azim1) == 1 or (azim0-azim1) == -(2**ring0-1):                                       #if neighbor in azimuth, retrograde
                        self.move_graph.add_weighted_edges_from([(nodes[0],nodes[1],FuelUsage[U.RETROGRADE])])     #add weighted edge, should look up fuel costs rather than hard code
                    elif (azim0-azim1) == -1 or (azim0-azim1) ==  2**ring0-1:
                        self.move_graph.add_weighted_edges_from([(nodes[0],nodes[1],FuelUsage[U.PROGRADE])])     
                    elif azim0-azim1 == 0:
                        self.move_graph.add_weighted_edges_from([(nodes[0],nodes[1],FuelUsage[U.NOOP])])

                if ring1 - ring0 == 1:                              #if node 1 is above node 0
                    if 2*azim0 == azim1:                            #rightmost sector above current sector
                        self.move_graph.add_weighted_edges_from([(nodes[0],nodes[1],FuelUsage[U.RADIAL_OUT])])
                if ring1 - ring0 == -1:                              #if node0 is above node 1
                    if floor(azim0/2) == azim1:
                        self.move_graph.add_weighted_edges_from([(nodes[0],nodes[1],FuelUsage[U.RADIAL_IN])])
              
                # turn graph, accounting for both drift and movement, assume object at node[0] to node[1]
                # useful for planning motion across board
                if ring0 == ring1:                                                                                 #if in same ring
                    if (azim0-azim1) == -2 or (azim0-azim1) == (2**ring0-2):                                                       #if two sector away in azimuth, prograde
                        self.turn_graph.add_weighted_edges_from([(nodes[0],nodes[1],FuelUsage[U.PROGRADE]+FuelUsage[U.DRIFT])])     #add weighted edge, should look up fuel costs rather than hard code
                    elif (azim0-azim1) == -1 or (azim0-azim1) ==  (2**ring0-1):                                                       #if adjacent node in drift direction
                        self.turn_graph.add_weighted_edges_from([(nodes[0],nodes[1],FuelUsage[U.NOOP]+FuelUsage[U.DRIFT])])     
                    elif azim0-azim1 == 0:                                                                                          #same sector
                        self.turn_graph.add_weighted_edges_from([(nodes[0],nodes[1],FuelUsage[U.RETROGRADE]+FuelUsage[U.DRIFT])])   #move retrograde and then drift back

                if ring1 - ring0 == 1:                              #if node 1 is above node 0
                    if 2*azim0+1 == azim1:                            #left sector above current sector
                        self.turn_graph.add_weighted_edges_from([(nodes[0],nodes[1],FuelUsage[U.RADIAL_OUT]+FuelUsage[U.DRIFT])])
                if ring1 - ring0 == -1:                              #if node0 is above node 1
                    if floor(azim0/2)+1 == azim1:
                        self.turn_graph.add_weighted_edges_from([(nodes[0],nodes[1],FuelUsage[U.RADIAL_IN]+FuelUsage[U.DRIFT])])





    def adjacent(self, sector1,sector2):

        ''' Check if two sectors are neighbors on board

        Args: two sectors
        Return: boolean value indicating neighborhood relationship on graph

        '''

        #since directed graph, neighborhood relationship should be check from both nodes
        neighbor_set = self.adj_graph[sector1]
        neighbor_status = sector2 in neighbor_set 

        return neighbor_status


            




