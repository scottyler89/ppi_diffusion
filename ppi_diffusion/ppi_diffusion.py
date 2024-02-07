import torch
import numpy as np
from scipy.sparse import lil_matrix as lil_mat
from ppi_diffusion.common_functions import read_dict
from ppi_diffusion.process_string import SYMBOL_INTERACT_FILE


class multi_network_diffuser():
    # Weave them together
    def __init__(self):
        pass



class single_diffuser():
    def __init__(self):
        pass



class diffuser(object):
    def __init__(self,
                 symbols,
                 interaction_mat):
        self.symbols = symbols
        self.symbol_hash = {v:k for k, v in enumerate(symbols)}
        self.interaction_mat = interaction_mat
    #
    #
    def get_symbols(self):
        return(self.symbols)
    #
    #
    def get_final_indices(self,
                          include_nodes):
        # If they did not specify a subset, do it on the full ppi
        if include_nodes is None:
            include_nodes = self.symbols
        # Filter for include_nodes
        include_nodes_final = [_ for _ in include_nodes if _ in self.symbol_hash]
        if len(include_nodes_final)<len(include_nodes):
            raise Warning("Some of your input nodes didn't map to the reference databse ("+str(len(include_nodes_final))+" out of "+str(len(include_nodes))+").\nYou can see what's in the database by running diffuser.get_symbols()")
        if len(include_nodes)==0:
            raise Exception("Didn't find any usable nodes in the include_nodes")
        include_idxs = [self.symbol_hash[_] for _ in include_nodes_final]
        include_symbols = [self.symbols[_] for _ in include_idxs]
        return(include_idxs, include_symbols)
    #
    #
    def get_final_seed_value_dict(self, seed_value_dict):
        present_keys = [k for k in seed_value_dict.keys() if k in self.symbol_hash]
        not_present_keys = [k for k in seed_value_dict.keys() if k not in self.symbol_hash]
        if len(not_present_keys)>0:
            raise Warning("Found some seeds that weren't in the reference ("+str(len(not_present_keys))+"). You can see what's in the database by running diffuser.get_symbols()")
        if len(present_keys)==0:
            raise Warning("Didn't find ANY of your seed values in our reference.\nYou can see what's in the database by running diffuser.get_symbols()")
        final_seed_value_dict = {k:seed_value_dict[k] for k in present_keys}
        return final_seed_value_dict
    #
    #
    def prep_diffuser(self,
                      seed_value_dict,
                      include_nodes = None,
                      degree_norm = True):
        # Single diffusor is where we log the results from this run & is what we will return
        diffused = single_diffuser()
        # run the real run
        include_idxs, include_symbols = self.get_final_indices(include_nodes)
        diffused.include_idxs = include_idxs
        diffused.include_symbols = include_symbols
        seed_value_dict = self.get_final_seed_value_dict(seed_value_dict)
        diffused.seed_value_dict = seed_value_dict
        # Subset the interaction matrix for only the ones that are pertinent
        used_interaction_mat = self.interaction_mat[include_idxs,:]
        used_interaction_mat = used_interaction_mat[:,include_idxs]
        diffused.used_interaction_mat = used_interaction_mat
        if degree_norm:
            # Calculate the degree of each node on the subnetwork that's included
            diffused.degree_vect = used_interaction_mat.sum(1)
        diffused.diffusion_matrix = torch.zeros((len(include_idxs),len(include_idxs)))
        ## Get the seed tensor
        seed_tensor = self.get_initial_seed_tensor(
                                include_idxs,
                                seed_value_dict,
                                do_shuffle = False,
                                degree_norm = True,
                                degree_vect = None)
        return diffused
    #
    #
    def run_diffusion(self,
                      seed_value_dict,
                      include_nodes = None,
                      degree_norm = True,
                      seed = 123456):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    #
    #
    def get_initial_seed_tensor(self,
                                include_idxs,
                                seed_value_dict,
                                do_shuffle = False,
                                degree_norm = True,
                                degree_vect = None):
        if degree_norm:
            assert degree_vect is not None
        seed_tensor = torch.zeros((1,len(include_idxs)))
        if not do_shuffle:
            # plug in the real values where the by belong
            for symbol, val in seed_value_dict.items():
                seed_tensor[self.symbol_hash[symbol]] = val
        else:
            # Now we do the shuffling instead
            idxs = np.arange(len(include_idxs))
            np.random.shuffle(idxs)
            all_vals = [val for symbol, val in seed_value_dict.items()]
            for i in range(len(all_vals)):
                seed_tensor[idxs[i]]=all_vals[i]
        return seed_tensor
    #
    #
    def run_diff_round(self,
                       seed_value_dict,
                       include_idxs, 
                       include_symbols,
                       used_interaction_mat,
                       do_shuffle = False,
                       degree_norm = True):
        diffused = self.prep_diffuser(seed_value_dict,
                      include_nodes = None,
                      degree_norm = True)
        if degree_norm:
            degree_vect
        

        return


def make_diffuser():
    symbols, interactions = read_dict(SYMBOL_INTERACT_FILE)
    return(diffuser(symbols, interactions))




