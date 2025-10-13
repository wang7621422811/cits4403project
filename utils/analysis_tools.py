"""
Network Analysis Tools for Beijing Subway System

This module provides various functions for analyzing subway network properties
including centrality measures, connectivity metrics, and structural analysis.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict


def calculate_network_stats(graph: nx.Graph) -> Dict[str, int | float]:
    """
    Calculate basic network statistics for a subway graph.
    
    Args:
        graph (nx.Graph): NetworkX graph representing the subway network
        
    Returns:
        Dict[str, int | float]: Dictionary containing basic network statistics
    """
    if graph.number_of_nodes() == 0:
        return {
            'num_nodes': 0, # the number of the notes
            'num_edges': 0,
            'density': 0.0,
            'num_components': 0,
            'largest_component_size': 0,
            'average_degree': 0.0,
            'max_degree': 0,
            'min_degree': 0
        }
    
    # Basic counts
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    # Degree statistics
    degrees = dict(graph.degree())
    degree_values = list(degrees.values())
    
    # Connected components
    components = list(nx.connected_components(graph))
    num_components = len(components)
    largest_component_size = len(max(components, key=len)) if components else 0
    
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': nx.density(graph),
        'num_components': num_components,
        'largest_component_size': largest_component_size,
        'average_degree': np.mean(degree_values),
        'max_degree': max(degree_values),
        'min_degree': min(degree_values),
        'degree_std': np.std(degree_values)
    }
    
    return stats


def calculate_average_shortest_path(graph: nx.Graph, 
                                  weight: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate average shortest path length for the network.
    
    Args:
        graph (nx.Graph): NetworkX graph representing the subway network
        weight (Optional[str]): Edge attribute to use as weight for path calculation
        
    Returns:
        Dict[str, float]: Dictionary containing path length statistics
    """
    if graph.number_of_nodes() == 0:
        return {
            'average_shortest_path': float('inf'),
            'diameter': float('inf'),
            'radius': float('inf'),
            'connectivity_ratio': 0.0
        }
    
    # Check if graph is connected
    if not nx.is_connected(graph):
        # For disconnected graphs, analyze the largest component
        largest_component = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_component)
        connectivity_ratio = len(largest_component) / graph.number_of_nodes()
    else:
        subgraph = graph
        connectivity_ratio = 1.0
    
    if subgraph.number_of_nodes() <= 1:
        return {
            'average_shortest_path': 0.0,
            'diameter': 0.0,
            'radius': 0.0,
            'connectivity_ratio': connectivity_ratio
        }
    
    try:
        # Calculate shortest path lengths
        if weight:
            avg_path_length = nx.average_shortest_path_length(subgraph, weight=weight)
            diameter = nx.diameter(subgraph, weight=weight)
            radius = nx.radius(subgraph, weight=weight)
        else:
            avg_path_length = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)
            radius = nx.radius(subgraph)
        
        return {
            'average_shortest_path': avg_path_length,
            'diameter': diameter,
            'radius': radius,
            'connectivity_ratio': connectivity_ratio
        }
    
    except nx.NetworkXError as e:
        return {
            'average_shortest_path': float('inf'),
            'diameter': float('inf'),
            'radius': float('inf'),
            'connectivity_ratio': connectivity_ratio,
            'error': str(e)
        }


def find_top_central_nodes(graph: nx.Graph, 
                          n: int = 5, 
                          centrality_type: str = 'degree') -> Dict[str, List[Tuple[str, float]]]:
    """
    Find the top n most central nodes using various centrality measures.
    
    Args:
        graph (nx.Graph): NetworkX graph representing the subway network
        n (int): Number of top nodes to return
        centrality_type (str): Type of centrality to calculate 
                              ('degree', 'betweenness', 'closeness', 'eigenvector', 'all')
        
    Returns:
        Dict[str, List[Tuple[str, float]]]: Dictionary with centrality type as key 
                                           and list of (node, centrality_value) tuples as value
    """
    if graph.number_of_nodes() == 0:
        return {}
    
    results = {}
    
    def get_top_n(centrality_dict: Dict[str, float], n: int) -> List[Tuple[str, float]]:
        """Helper function to get top n nodes from centrality dictionary."""
        sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:min(n, len(sorted_nodes))]
    
    if centrality_type in ['degree', 'all']:
        degree_centrality = nx.degree_centrality(graph)
        results['degree'] = get_top_n(degree_centrality, n)
    
    if centrality_type in ['betweenness', 'all']:
        try:
            betweenness_centrality = nx.betweenness_centrality(graph)
            results['betweenness'] = get_top_n(betweenness_centrality, n)
        except:
            results['betweenness'] = []
    
    if centrality_type in ['closeness', 'all']:
        try:
            # Only calculate for connected components
            if nx.is_connected(graph):
                closeness_centrality = nx.closeness_centrality(graph)
                results['closeness'] = get_top_n(closeness_centrality, n)
            else:
                # Calculate for largest component
                largest_component = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_component)
                closeness_centrality = nx.closeness_centrality(subgraph)
                results['closeness'] = get_top_n(closeness_centrality, n)
        except:
            results['closeness'] = []
    
    if centrality_type in ['eigenvector', 'all']:
        try:
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
            results['eigenvector'] = get_top_n(eigenvector_centrality, n)
        except:
            results['eigenvector'] = []
    
    return results


def calculate_clustering_metrics(graph: nx.Graph) -> Dict[str, float]:
    """
    Calculate clustering-related metrics for the network.
    
    Args:
        graph (nx.Graph): NetworkX graph representing the subway network
        
    Returns:
        Dict[str, float]: Dictionary containing clustering metrics
    """
    if graph.number_of_nodes() == 0:
        return {
            'average_clustering': 0.0,
            'global_clustering': 0.0,
            'transitivity': 0.0
        }
    
    try:
        avg_clustering = nx.average_clustering(graph)
        transitivity = nx.transitivity(graph)
        
        # Global clustering coefficient (alternative calculation)
        triangles = sum(nx.triangles(graph).values()) / 3
        possible_triangles = sum([d * (d - 1) / 2 for n, d in graph.degree()])
        global_clustering = triangles / possible_triangles if possible_triangles > 0 else 0.0
        
        return {
            'average_clustering': avg_clustering,
            'global_clustering': global_clustering,
            'transitivity': transitivity
        }
    
    except Exception as e:
        return {
            'average_clustering': 0.0,
            'global_clustering': 0.0,
            'transitivity': 0.0,
            'error': str(e)
        }


def analyze_network_efficiency(graph: nx.Graph) -> Dict[str, float]:
    """
    Calculate network efficiency metrics.
    
    Args:
        graph (nx.Graph): NetworkX graph representing the subway network
        
    Returns:
        Dict[str, float]: Dictionary containing efficiency metrics
    """
    if graph.number_of_nodes() <= 1:
        return {
            'global_efficiency': 0.0,
            'local_efficiency': 0.0,
            'nodal_efficiency_avg': 0.0
        }
    
    try:
        # Global efficiency
        global_eff = nx.global_efficiency(graph)
        
        # Local efficiency
        local_eff = nx.local_efficiency(graph)
        
        # Average nodal efficiency
        nodal_efficiencies = []
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if len(neighbors) > 1:
                subgraph = graph.subgraph(neighbors)
                if subgraph.number_of_edges() > 0:
                    nodal_eff = nx.global_efficiency(subgraph)
                    nodal_efficiencies.append(nodal_eff)
        
        nodal_eff_avg = np.mean(nodal_efficiencies) if nodal_efficiencies else 0.0
        
        return {
            'global_efficiency': global_eff,
            'local_efficiency': local_eff,
            'nodal_efficiency_avg': nodal_eff_avg
        }
    
    except Exception as e:
        return {
            'global_efficiency': 0.0,
            'local_efficiency': 0.0,
            'nodal_efficiency_avg': 0.0,
            'error': str(e)
        }


def find_critical_nodes(graph: nx.Graph, n: int = 5) -> Dict[str, List[Tuple[str, float]]]:
    """
    Identify critical nodes in the network based on various criteria.
    
    Args:
        graph (nx.Graph): NetworkX graph representing the subway network
        n (int): Number of critical nodes to return for each criterion
        
    Returns:
        Dict[str, List[Tuple[str, float]]]: Dictionary with criterion as key 
                                           and list of (node, score) tuples as value
    """
    if graph.number_of_nodes() == 0:
        return {}
    
    results = {}
    
    # Nodes with highest degree (most connections)
    degree_dict = dict(graph.degree())
    results['highest_degree'] = sorted(degree_dict.items(), 
                                     key=lambda x: x[1], reverse=True)[:n]
    
    # Nodes whose removal would most increase average shortest path
    if nx.is_connected(graph):
        original_avg_path = nx.average_shortest_path_length(graph)
        path_impact = {}
        
        for node in graph.nodes():
            temp_graph = graph.copy()
            temp_graph.remove_node(node)
            
            if temp_graph.number_of_nodes() > 0 and nx.is_connected(temp_graph):
                new_avg_path = nx.average_shortest_path_length(temp_graph)
                path_impact[node] = new_avg_path - original_avg_path
            else:
                path_impact[node] = float('inf')  # Disconnects the graph
        
        results['path_impact'] = sorted(path_impact.items(), 
                                      key=lambda x: x[1], reverse=True)[:n]
    
    # Nodes whose removal would create most components
    component_impact = {}
    original_components = nx.number_connected_components(graph)
    
    for node in graph.nodes():
        temp_graph = graph.copy()
        temp_graph.remove_node(node)
        new_components = nx.number_connected_components(temp_graph)
        component_impact[node] = new_components - original_components
    
    results['component_impact'] = sorted(component_impact.items(), 
                                       key=lambda x: x[1], reverse=True)[:n]
    
    return results


def calculate_small_world_metrics(graph: nx.Graph) -> Dict[str, float]:
    """
    Calculate small-world network metrics.
    
    Args:
        graph (nx.Graph): NetworkX graph representing the subway network
        
    Returns:
        Dict[str, float]: Dictionary containing small-world metrics
    """
    if graph.number_of_nodes() < 4:
        return {
            'clustering_coefficient': 0.0,
            'average_path_length': 0.0,
            'small_world_sigma': 0.0,
            'small_world_omega': 0.0
        }
    
    try:
        # Calculate actual metrics
        C = nx.average_clustering(graph)
        
        if nx.is_connected(graph):
            L = nx.average_shortest_path_length(graph)
        else:
            # Use largest component
            largest_component = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_component)
            L = nx.average_shortest_path_length(subgraph) if subgraph.number_of_nodes() > 1 else 0
        
        # Generate random graph with same degree sequence for comparison
        try:
            degree_sequence = [d for n, d in graph.degree()]
            random_graph = nx.configuration_model(degree_sequence)
            random_graph = nx.Graph(random_graph)  # Remove multi-edges and self-loops
            random_graph.remove_edges_from(nx.selfloop_edges(random_graph))
            
            C_rand = nx.average_clustering(random_graph)
            if nx.is_connected(random_graph):
                L_rand = nx.average_shortest_path_length(random_graph)
            else:
                largest_comp = max(nx.connected_components(random_graph), key=len)
                subgraph_rand = random_graph.subgraph(largest_comp)
                L_rand = nx.average_shortest_path_length(subgraph_rand) if subgraph_rand.number_of_nodes() > 1 else L
            
            # Small-world metrics
            sigma = (C / C_rand) / (L / L_rand) if C_rand > 0 and L_rand > 0 else 0
            omega = (L_rand / L) - (C / C_rand) if C_rand > 0 and L > 0 else 0
            
        except:
            sigma = 0.0
            omega = 0.0
        
        return {
            'clustering_coefficient': C,
            'average_path_length': L,
            'small_world_sigma': sigma,
            'small_world_omega': omega
        }
    
    except Exception as e:
        return {
            'clustering_coefficient': 0.0,
            'average_path_length': 0.0,
            'small_world_sigma': 0.0,
            'small_world_omega': 0.0,
            'error': str(e)
        }


def analyze_degree_distribution(graph: nx.Graph) -> Dict[str, List | float]:
    """
    Analyze the degree distribution of the network.
    
    Args:
        graph (nx.Graph): NetworkX graph representing the subway network
        
    Returns:
        Dict[str, List | float]: Dictionary containing degree distribution analysis
    """
    if graph.number_of_nodes() == 0:
        return {
            'degree_sequence': [],
            'degree_histogram': [],
            'power_law_alpha': None,
            'degree_entropy': 0.0
        }
    
    # Get degree sequence
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    
    # Create degree histogram
    degree_count = defaultdict(int)
    for degree in degree_sequence:
        degree_count[degree] += 1
    
    degrees = sorted(degree_count.keys())
    counts = [degree_count[d] for d in degrees]
    
    # Calculate degree entropy
    total_nodes = sum(counts)
    probabilities = [c / total_nodes for c in counts]
    degree_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    # Attempt to fit power law (simplified)
    try:
        if len(degree_sequence) > 1 and max(degree_sequence) > 1:
            # Simple power law fitting using log-log regression
            degrees_nonzero = [d for d in degree_sequence if d > 0]
            if len(degrees_nonzero) > 1:
                log_degrees = np.log(degrees_nonzero)
                log_probs = np.log(np.arange(1, len(degrees_nonzero) + 1) / len(degrees_nonzero))
                alpha = -np.polyfit(log_degrees, log_probs, 1)[0]
            else:
                alpha = None
        else:
            alpha = None
    except:
        alpha = None
    
    return {
        'degree_sequence': degree_sequence,
        'degree_histogram': list(zip(degrees, counts)),
        'power_law_alpha': alpha,
        'degree_entropy': degree_entropy,
        'max_degree': max(degree_sequence) if degree_sequence else 0,
        'min_degree': min(degree_sequence) if degree_sequence else 0,
        'median_degree': np.median(degree_sequence) if degree_sequence else 0
    }
