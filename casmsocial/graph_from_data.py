import getopt
import math
import sys

import pandas as pd
from loguru import logger


def isNumber(v):
    return isinstance(v, (int, float))


def map_spid_to_graph_id(places_f):
    map_df = pd.DataFrame()

    for f in places_f:
        df = pd.read_csv(f)

        df = df["sp_id"]
        logger.debug(len(df))

        map_df = pd.concat([map_df, df], axis=0, ignore_index=True)

    map_df.columns = ["sp_id"]
    map_df["graph_id"] = range(len(map_df))
    map_df["graph_id"] = map_df["graph_id"] + 1

    ret_map = dict(zip(map_df["sp_id"], map_df["graph_id"]))
    return ret_map


def add_to_dict(key, val, d):
    if key is not None and val is not None:
        if key in d:
            d[key].append(val)
        else:
            d[key] = [val]
    return d


def doubly_add_to_dict(val1, val2, d):
    d = add_to_dict(val1, val2, d)
    d = add_to_dict(val2, val1, d)
    return d


def full_map_of_adjacent_places(person_f):
    full_E = {}
    persons = pd.read_csv(person_f)

    for hh, school, work in zip(persons["sp_hh_id"], persons["sp_school_id"], persons["sp_work_id"]):
        hh_id = None if math.isnan(hh) else int(hh)
        school_id = None if math.isnan(school) else int(school)
        work_id = None if math.isnan(work) else int(work)

        full_E = doubly_add_to_dict(hh_id, school_id, full_E)
        full_E = doubly_add_to_dict(hh_id, work_id, full_E)
        full_E = doubly_add_to_dict(work_id, school_id, full_E)

    return full_E


def trim_map(edges):
    for v, adjacent in edges.items():
        for a_v in adjacent:
            try:
                edges[a_v].remove(v)
            except ValueError:
                continue
    return edges


def get_num_v_and_e(edges):
    num_v = len(edges)
    num_e = 0
    for _v, e in edges.items():
        num_e = num_e + len(e)

    if num_e % 2 != 0:
        logger.debug("Uneven number of edges??!!")

    num_e = num_e / 2
    return num_v, int(num_e)


def make_spid_to_graphid_map(edges):
    spid_to_graphid = {}
    i = 1
    for v in edges:
        spid_to_graphid[v] = i
        i = i + 1

    return spid_to_graphid


def write_graph_file(output_f, num_v, num_e, edges, spid_to_graphid):
    with open(output_f, "w") as f:
        f.write(f"{num_v} {num_e}\n")
        for _v, adjacent in edges.items():
            f.write(" ".join([str(spid_to_graphid[node]) for node in adjacent]))
            f.write("\n")


def write_map_file(output_f, spid_to_graphid):
    with open(output_f, "w") as f:
        for k, v in spid_to_graphid.items():
            f.write(f"{k} {v}\n")


def print_help():
    print("Options:\n-a|--agent_file <agent_filename>")
    print("-o|--output_file <output_graph_filename>\n-m|--map_file <id_mappings_filename")
    print("\nExample:\npython graph_from_data.py -a data/persons.csv -o graph.txt -m graph_id_map.txt")


if __name__ == "__main__":
    person_f = "data/persons.csv"
    output_f = "data/graph.txt"
    output_map_f = "data/graph_id_map.txt"

    try:
        opts, args = getopt.getopt(sys.argv[1:], "ha:o:m:", ["agent_file=", "output_file=", "map_file="])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print_help()
            sys.exit()
        elif opt in ("-a", "--agent_file"):
            person_f = arg
        elif opt in ("-o", "--output_file"):
            output_f = arg
        elif opt in ("-m", "--map_file"):
            output_map_f = arg

    full_E = full_map_of_adjacent_places(person_f)

    num_v, num_e = get_num_v_and_e(full_E)
    spid_to_graphid_map = make_spid_to_graphid_map(full_E)
    write_graph_file(output_f, num_v, num_e, full_E, spid_to_graphid_map)
    write_map_file(output_map_f, spid_to_graphid_map)

    logger.debug(f"NumV: {num_v}, NumE: {num_e}")
