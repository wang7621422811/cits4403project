# Beijing Subway Network Evolution Analysis

This project analyzes how Beijing's subway network has developed over the past 50 years using graph theory and network analysis. We examine the structural changes from 1969 to 2023 to understand how one of the world's largest metro systems evolved.

## Project Overview

Beijing's subway system has grown dramatically from just 8 stations in 1969 to over 321 stations today. This project uses computational modeling to study this transformation. We apply graph theory to analyze the network structure at different time periods and identify important patterns in how the system developed.

### What We Did

Our analysis focuses on four main areas:

1. **Network Growth Analysis** - We tracked how the network size, connectivity, and efficiency changed over time
2. **Hub Station Evolution** - We identified which stations became important transfer points and how this changed
3. **Network Topology** - We visualized how the network structure transformed from simple lines to a complex interconnected system
4. **Scenario Testing** - We tested what happens if key stations are removed or new stations are added

### Key Findings

- The network grew by nearly 4,000% in station count but became more efficient through better design
- Hub stations shifted from a few critical points to multiple distributed transfer stations, making the system more reliable
- Adding individual stations has minimal impact on overall efficiency, but strategic line additions can significantly improve the network

## Setup Instructions

### Requirements

You need Python 3.8 or higher installed on your computer. If you don't have Python, download it from [python.org](https://www.python.org/downloads/).

### Installation Steps

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/wang7621422811/CITS4403-44-Project.git
   cd CITS4403-44-Project
   ```

2. **Create a virtual environment** (recommended but optional)
   ```bash
   # On Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install all necessary libraries including pandas, networkx, matplotlib, and jupyter.

4. **Verify installation**
   ```bash
   python -c "import networkx; import pandas; print('Installation successful!')"
   ```

If you see "Installation successful!", you're ready to go!

## Usage Guide

### Quick Start

The easiest way to explore our analysis is through Jupyter notebooks. Here's how to get started:

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   This will open a browser window showing the project files.

2. **Open a notebook**
   Navigate to the `notebooks/` folder and open any of these files:
   
   - `analysis.ipynb` - Main temporal analysis (start here!)
   - `hub_evolution_analysis.ipynb` - Hub station analysis
   - `network_topology_visualization.ipynb` - Network structure visualizations
   - `scenario_analysis.ipynb` - What-if scenario testing

3. **Run the analysis**
   - Click on a cell and press `Shift + Enter` to run it
   - Or use the "Run All" button to execute the entire notebook
   - Results and figures will appear below each code cell

### Using the Code Directly

If you prefer to use the code in your own scripts:

```python
# Import the subway model
from src.subway_model import SubwayGraph

# Load the data
subway = SubwayGraph('data/beijing_subway_data.csv')

# Build a network for a specific year
graph_2023 = subway.build_graph_for_date('2023')

# Get basic statistics
print(f"Stations: {graph_2023.number_of_nodes()}")
print(f"Connections: {graph_2023.number_of_edges()}")

# Use analysis tools
from utils.analysis_tools import calculate_network_stats

stats = calculate_network_stats(graph_2023)
print(f"Network density: {stats['density']:.4f}")
```

### Generating Figures

All figures are automatically generated when you run the notebooks. They will be saved in the `figures/` folder. You can use these images in reports or presentations.

### Analyzing Different Time Periods

To analyze a different year:

```python
# Any year between 1969 and 2023
graph_2008 = subway.build_graph_for_date('2008')
graph_1990 = subway.build_graph_for_date('1990')
```

The model will automatically include only stations that were open by that date.

## Project Structure

```
CITS4403-44-Project/
├── src/
│   └── subway_model.py          # Main model class for building network graphs
├── utils/
│   └── analysis_tools.py        # Functions for calculating network metrics
├── data/
│   └── beijing_subway_data.csv  # Historical station and connection data
├── notebooks/
│   ├── analysis.ipynb                          # Main temporal analysis
│   ├── hub_evolution_analysis.ipynb            # Hub station evolution
│   ├── network_topology_visualization.ipynb    # Network structure plots
│   └── scenario_analysis.ipynb                 # What-if scenarios
├── figures/                     # Generated visualizations (created when running notebooks)
├── requirements.txt             # Python package dependencies
├── PROJECT_REPORT.md           # Detailed academic report
└── README.md                   # This file
```

## Understanding the Results

### Network Metrics Explained

- **Nodes/Stations**: Number of subway stations in the network
- **Edges/Connections**: Direct links between adjacent stations
- **Density**: How interconnected the network is (0 = no connections, 1 = fully connected)
- **Average Path Length**: Average number of stations between any two points
- **Centrality**: Measures how important a station is in the network

### What the Visualizations Show

- **Temporal Evolution Charts**: Show how metrics changed over time
- **Network Topology Maps**: Display the actual structure of the network at different periods
- **Hub Evolution Charts**: Compare which stations were most important in different years
- **Scenario Analysis**: Demonstrate the impact of network changes

## Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'networkx'`
**Solution**: Make sure you installed the requirements: `pip install -r requirements.txt`

**Problem**: Jupyter notebook won't start
**Solution**: Install jupyter explicitly: `pip install jupyter`

**Problem**: Figures not displaying in notebook
**Solution**: Add this line at the start of your notebook: `%matplotlib inline`

**Problem**: Data file not found
**Solution**: Make sure you're running commands from the project root directory

## Contributing

This project was developed for CITS4403 Computational Modelling. If you find issues or have suggestions, feel free to open an issue on GitHub.

## License

This project is for educational purposes as part of university coursework.

## Contact

For questions about this project, please contact the project team through the GitHub repository.

## Acknowledgments

- Beijing subway data compiled from official transit authority records
- Analysis framework built using NetworkX library
- Visualization tools provided by Matplotlib and Seaborn

---

**Note**: This is an academic project demonstrating computational modeling techniques. The analysis focuses on network structure rather than operational aspects like passenger flow or service frequency.