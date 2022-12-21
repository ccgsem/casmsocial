# Using METIS for Graph Partitioning with communitysim

## Overview

METIS is a set of serial programs for partitioning graphs, partitioning finite element meshes, and producing fill reducing orderings for sparse matrices. The download, documentation, etc. can be found at the [METIS Homepage](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview). The goal for dcSIM is to partition agents between processes such that the least amount of inter-process communication is required. To do this, a graph is created, `G(V, E)`, with vertices, `V`, as places and edges, `E`, as people.

Example: If a Person is scheduled to move between places `A`, `B`, and `C`, then the following edges will be created.
```
A - B
A - C
B - C
```

The edges are undirected. See below for instructions on creating the graph and partitioning it.

## Creating the Graph

To create the graph, run the provided `graph_from_data.py` python script in the `scripts` directory. This will use the agent input file to create two output files.
1. Output Graph: This is the graph that will be partitioned by METIS.
2. Output ID mappings: These mappings will be used by the final script to map process assignments back to the agents and places.

Run `python graph_from_data.py -h` to see command line argument options.

## Partitioning the Graph

First you must have METIS installed. Follow the instructions at the [METIS Download page](http://glaros.dtc.umn.edu/gkhome/metis/metis/download).

You can read through the manual to get a more in depth view of METIS, but for our case, we will simply want to use the `gpmetis` program. (See page 13 of the manual.)

Run `gpmetis <graph_filename> <num_partitions>`.

The output file from that will be called `<graph_filename>.part.<num_partitions>`

## Adding Partition Data to Input Files

The final step is to add the process assignment to the input files. For that we have a second python script, `processes_from_partition.py`, also in the `scripts` directory. This script will take in the agent/place input files, the process assignments from METIS, and the ID mappings from `graph_from_data.py`. It will create copies of the agent/place input files with `'_partition'` added to the filename and a new column with the process assignment.

## Using the Partition Assignments

Simply make sure you are on a branch that supports using process assignments directly during setup instead of calculating process from ward. If you need to add this support to another branch, you will need to modify `ModelSetup.h` to use the `'process'` column for process assignments instead of calculating from the ward column.