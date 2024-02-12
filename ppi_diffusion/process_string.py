import os
import pandas as pd
import numpy as np
from gprofiler import GProfiler
from scipy.sparse import lil_matrix as lil_mat
import random
from ppi_diffusion.common_functions import save_dict

PACKAGE_DIR = os.path.dirname(__file__)
STRINGDB_INTERACT_FILE = os.path.join(PACKAGE_DIR,"symbol_interaction_lilmat.pkl")


def strip_species(in_vect):
    return([thing.split(".")[1] for thing in in_vect])

def get_lookup(p1, p2):
    p = list(set(p1).union(set(p2)))
    # First map to gene level
    gp = GProfiler('mapper_'+str(random.randint(0,int(1e6))), want_header=True)
    results = gp.gconvert(p, 
        organism = "hsapiens", target='ENSG')
    results = pd.DataFrame(results[1:], columns = results[0])
    out_dict = {}
    all_symbols = list(sorted(set([thing for thing in results["name"] if thing is not None])))
    symbol_hash = {v:k for k, v in enumerate(all_symbols)}
    for _, row in results.iterrows():
        if not row["name"]is None:
            if row["initial alias"] not in out_dict:
                out_dict[row["initial alias"]]=[]
            temp_list = out_dict[row["initial alias"]]
            temp_list.append(symbol_hash[row["name"]])
            out_dict[row["initial alias"]] = temp_list
    return(out_dict, all_symbols)


def get_symbol_interaction_mat(in_db):
    p_to_idx_dict, symbol_list = get_lookup(in_db.iloc[:,1],in_db.iloc[:,0].tolist())
    interaction_mat = lil_mat((len(symbol_list),len(symbol_list)),dtype=bool)
    for _, row in in_db.iterrows():
        if row["protein1"] in p_to_idx_dict and row["protein2"] in p_to_idx_dict:
            p1_idxs = p_to_idx_dict[row["protein1"]]
            p2_idxs = p_to_idx_dict[row["protein2"]]
            for i in p1_idxs:
                for j in p2_idxs:
                    interaction_mat[i,j]=True
                    interaction_mat[j,i]=True
    return(symbol_list, interaction_mat)


def load_and_process_stringdb(f_name = "9606.protein.physical.links.full.v12.0.txt.gz"):
    global PACKAGE_DIR, SYMBOL_INTERACT_FILE
    in_db = pd.read_csv(os.path.join(PACKAGE_DIR,f_name), sep=" ", compression="gzip")
    ## First two columns have the ensp ids
    for i in [0,1]:
        in_db.iloc[:,i]=strip_species(in_db.iloc[:,i])
    ## convert them to symbols
    save_dict(get_symbol_interaction_mat(in_db), SYMBOL_INTERACT_FILE)


if __name__=="__main__":
    load_and_process_stringdb()







