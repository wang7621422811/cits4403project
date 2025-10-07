"""
Beijing Subway Network Model

This module contains the SubwayGraph class for modeling and analyzing
the Beijing subway network using graph theory.
"""

import pandas as pd
import networkx as nx
from datetime import datetime
from typing import Optional, List, Dict, Set
import os


class SubwayGraph:
    """
    A class to represent and analyze the Beijing subway network.
    
    This class can build subway network graphs for different time periods
    based on station opening dates, allowing for temporal analysis of
    network evolution.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the SubwayGraph with subway data.
        
        Args:
            data_path (str): Path to the CSV file containing subway data
        """
        self.data_path = data_path
        self.data = None
        self.stations_info = {}  # Store station metadata
        self._load_data()
    
    def _load_data(self) -> None:
        """
        Load subway data from CSV file.
        
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data format is invalid
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        try:
            self.data = pd.read_csv(self.data_path)
            # Validate required columns
            required_columns = ['station_name', 'line', 'opening_date', 'connections']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert opening_date to datetime
            self.data['opening_date'] = pd.to_datetime(self.data['opening_date'])
            
            # Build stations info dictionary
            self._build_stations_info()
            
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
    
    def _build_stations_info(self) -> None:
        """
        Build a dictionary containing station information for quick lookup.
        """
        for _, row in self.data.iterrows():
            station_name = row['station_name']
            self.stations_info[station_name] = {
                'line': row['line'],
                'opening_date': row['opening_date'],
                'connections': self._parse_connections(row['connections'])
            }
    
    def _parse_connections(self, connections_str: str) -> List[str]:
        """
        Parse connection string into a list of connected stations.
        
        Args:
            connections_str (str): Semicolon-separated string of connected stations
            
        Returns:
            List[str]: List of connected station names
        """
        if pd.isna(connections_str) or connections_str.strip() == '':
            return []
        
        # Split by semicolon and clean up whitespace
        connections = [conn.strip() for conn in str(connections_str).split(';')]
        return [conn for conn in connections if conn]  # Remove empty strings
    
    def build_graph_for_date(self, target_date: str) -> nx.Graph:
        """
        Build a subway network graph for stations opened before or on the target date.
        
        Args:
            target_date (str): Target date in format 'YYYY-MM-DD' or 'YYYY'
            
        Returns:
            nx.Graph: NetworkX graph representing the subway network
            
        Raises:
            ValueError: If the date format is invalid
        """
        try:
            # Parse the target date
            if len(target_date) == 4:  # Year only
                target_datetime = datetime.strptime(f"{target_date}-12-31", "%Y-%m-%d")
            else:
                target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {target_date}. Use 'YYYY-MM-DD' or 'YYYY'")
        
        # Filter stations that were open by the target date
        valid_stations = self._get_stations_by_date(target_datetime)
        
        # Build the graph
        graph = nx.Graph()
        
        # Add nodes (stations) with attributes
        for station in valid_stations:
            station_info = self.stations_info[station]
            graph.add_node(station, 
                          line=station_info['line'],
                          opening_date=station_info['opening_date'])
        
        # Add edges (connections) between stations
        for station in valid_stations:
            connections = self.stations_info[station]['connections']
            for connected_station in connections:
                # Only add edge if both stations exist in the valid stations set
                if connected_station in valid_stations:
                    graph.add_edge(station, connected_station)
        
        return graph
    
    def _get_stations_by_date(self, target_date: datetime) -> Set[str]:
        """
        Get all stations that were open by the target date.
        
        Args:
            target_date (datetime): Target date
            
        Returns:
            Set[str]: Set of station names
        """
        valid_stations = set()
        
        for station, info in self.stations_info.items():
            if info['opening_date'] <= target_date:
                valid_stations.add(station)
        
        return valid_stations
    
    def get_network_stats(self, graph: nx.Graph) -> Dict[str, any]:
        """
        Calculate basic network statistics for a given graph.
        
        Args:
            graph (nx.Graph): NetworkX graph
            
        Returns:
            Dict[str, any]: Dictionary containing network statistics
        """
        if graph.number_of_nodes() == 0:
            return {
                'num_stations': 0,
                'num_connections': 0,
                'num_components': 0,
                'largest_component_size': 0,
                'average_degree': 0
            }
        
        stats = {
            'num_stations': graph.number_of_nodes(),
            'num_connections': graph.number_of_edges(),
            'num_components': nx.number_connected_components(graph),
            'largest_component_size': len(max(nx.connected_components(graph), key=len)),
            'average_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        }
        
        return stats
    
    def get_available_date_range(self) -> tuple:
        """
        Get the date range of available data.
        
        Returns:
            tuple: (earliest_date, latest_date) as datetime objects
        """
        if self.data is None or self.data.empty:
            return None, None
        
        earliest = self.data['opening_date'].min()
        latest = self.data['opening_date'].max()
        
        return earliest, latest
    
    def get_stations_by_line(self, line_number: int) -> List[str]:
        """
        Get all stations on a specific subway line.
        
        Args:
            line_number (int): Line number
            
        Returns:
            List[str]: List of station names on the specified line
        """
        stations = []
        for station, info in self.stations_info.items():
            if info['line'] == line_number:
                stations.append(station)
        
        return sorted(stations)
    
    def __str__(self) -> str:
        """String representation of the SubwayGraph object."""
        if self.data is None:
            return "SubwayGraph: No data loaded"
        
        earliest, latest = self.get_available_date_range()
        return (f"SubwayGraph: {len(self.stations_info)} stations, "
                f"data from {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
    
    def __repr__(self) -> str:
        """Detailed representation of the SubwayGraph object."""
        return f"SubwayGraph(data_path='{self.data_path}')"
