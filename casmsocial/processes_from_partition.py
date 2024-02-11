
import pandas as pd
import sys, getopt


def get_process(spid, partition_map, spid_to_node):
    if spid in spid_to_node:
        node = spid_to_node[spid]
        if node in partition_map:
            return partition_map[node]
    return 0

def add_process_to_places(places_f, partition_map, spid_to_node, augment):
    for f in places_f:
        df = pd.read_csv(f)
        processes = [get_process(spid, partition_map, spid_to_node) for spid in df['sp_id']]
        df['process'] = processes

        out_file = augmentFilename(f, augment)
        df.to_csv(out_file, index=False)

def add_process_to_persons(persons_f, partition_map, spid_to_node, augment):
    df = pd.read_csv(persons_f)
    processes = [get_process(spid, partition_map, spid_to_node) for spid in df['sp_hh_id']]
    df['process'] = processes

    out_file = augmentFilename(persons_f, augment)
    df.to_csv(out_file, index=False)


def read_partition_file(partition_f):
    with open(partition_f, 'r') as f:
        lines = f.readlines()
        return [int(p) for p in lines]

def partition_to_partition_map(partitions):
    return {i: partitions[i-1] for i in range(1, len(partitions)+1)}

def read_id_map_file(node_map_f):
    with open(node_map_f, 'r') as f:
        lines = f.readlines()
        id_map = {int(v[0]): int(v[1]) for v in [line.split() for line in lines]}
        return id_map

def invert_id_map(spid_to_node):
    node_to_spid = {v: k for k, v in spid_to_node.items()}
    return node_to_spid

def augmentFilename(f, augment):
    parts = f.split('.')
    
    ret = parts[0]
    if len(parts) > 2:
        for p in parts[1:-1]:
            ret = ret + '.' + p
    return ret + augment + '.' + parts[-1]

def print_help():
    print("Options:\n-a|--agent_file <agent_filename>\n-p|--place_file <place_file1> -p|--place_file <place_file2> ...")
    print("--partitions <partition_filename>\n-m|--map_file <id_mappings_filename")
    print("\nExample:\npython processes_from_partition.py -a data/persons.csv -p data/hh.csv |")
    print("\t-p data/work.csv -p data/schools.csv --partitions data/graph.txt.part.4 |")
    print("\t-m data/graph_id_map.txt")
    print("\nOutput:\nWill create copies of the agent and place files with '_partition' added to the filename.")


if __name__ == "__main__":
    places_f = []
    persons_f = 'data/persons.csv'
    partition_f = 'data/graph.txt.part.4'
    node_map_f = 'data/graph_id_map.txt'
    augment = '_partition'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "ha:p:m:",["agent_file=","place_file=","partitions=","map_file="])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt in ("-a","--agent_file"):
            person_f = arg
        elif opt in ("-p","--place_file"):
            places_f.append(arg)
        elif opt == "--partitions":
            partition_f = arg
        elif opt in ("-m","--map_file"):
            node_map_f = arg


    if len(places_f) == 0:
        places_f = ['data/hh.csv', 'data/work.csv', 'data/schools.csv']
    
    partitions = read_partition_file(partition_f)
    partition_map = partition_to_partition_map(partitions)
    spid_to_node = read_id_map_file(node_map_f)

    add_process_to_places(places_f, partition_map, spid_to_node, augment)
    add_process_to_persons(persons_f, partition_map, spid_to_node, augment)