import torch
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix as lil_mat
from ppi_diffusion.common_functions import read_dict
from ppi_diffusion.process_string import STRINGDB_INTERACT_FILE


class multi_network_diffuser():
    # Weave them together
    def __init__(self):
        pass



class single_diffuser():
    def __init__(self,
                 symbols,
                 used_interaction_mat,
                 seed_tensor):
        assert np.sum(np.isnan(seed_tensor.numpy())) == 0
        print("TOPTOP SEED:",seed_tensor)
        self.symbols = symbols
        self.interaction_tensor = torch.tensor(used_interaction_mat.todense().astype(np.float32))
        print(np.sum(np.abs(seed_tensor.numpy())))
        seed_tensor = seed_tensor/(np.sum(np.abs(seed_tensor.numpy()))/len(symbols))
        self.seed_tensor = torch.tensor(seed_tensor,dtype=torch.float32)
        print("TOP pre-norm SEED:",self.seed_tensor)
        print("MIDTOP SEED:",self.seed_tensor)
        self.finished = False
        self.network_prepped = False
    #
    #
    def run_diffusions(self,
                       n_steps = 50,
                      n_perm = 50):
        # Run the shuffled null diffusions
        self.null_diffused = torch.zeros((n_perm,len(self.symbols)))
        for i in range(n_perm):
            rand_seed = self.seed_tensor.clone()[:,torch.randperm(len(self.symbols))]
            temp_null=self.diffuse(
                n_steps,
                rand_seed,
                return_steps=False
            )
            self.null_diffused[i,:] = temp_null
        # run the real diffusion experiment first
        ## Now get the Z-scores
        means = self.null_diffused.mean(0)
        sds = self.null_diffused.std(0)
        self.diffused_values, self.diffusion_steps = self.diffuse(n_steps,
                                                                  self.seed_tensor,
                                                                  return_steps = True,
                                                                  null_means = means)
        #self.standardized_residuals = self.diffused_values - means
        #self.standardized_residuals /= sds
        #self.standardized_steps = self.diffusion_steps - means
        #self.standardized_steps /= sds
        self.node_variability = self.diffusion_steps.std(0)
        self.finished=True
    #
    #
    def diffuse(self,
                n_steps,
                temp_seed,
                return_steps = True,
                null_means = None):
        if null_means is None:
            null_means = torch.ones_like(temp_seed)
        if return_steps:
            # rows: steps
            # cols: symbols
            steps = np.zeros((n_steps+1, 
                                 len(self.symbols)))
            print(temp_seed.detach().clone().numpy().shape)
            print(steps.shape)
            steps[0,:]=temp_seed.detach().clone().numpy()
        for i in range(n_steps):
            # do the diffusion
            print("top_step",temp_seed)
            temp_seed = torch.matmul(temp_seed,self.interaction_tensor)
            # normalize to sum
            temp_seed /= temp_seed.abs().sum()/len(self.symbols)
            # normalize to null
            temp_seed = (1+temp_seed) / (1+null_means)
            # renormalize to sum
            temp_seed /= temp_seed.abs().sum()/len(self.symbols)
            if return_steps:
                steps[i+1,:]=temp_seed.detach().clone().numpy()
        if return_steps:
            return (temp_seed, steps)
        else:
            return (temp_seed)
    #
    #
    def plot_steps(self):
        if not self.finished:
            print("You'll need to run the diffusion first: diffuser.run_diffusions(n_steps=n, n_perm = np)")
        else:
            sns.clustermap(self.diffused_values,
                           row_cluster = False,
                           metric="cosine")
            plt.show()
    #
    #
    def prep_network(self):
        # TODO
        self.network_prepped = True
        self.G = nx.from_numpy_array(self.interaction_tensor.numpy())
        nx.relabel_nodes(self.G, {i:self.symbols[i] for i in np.arange(len(self.symbols))}, copy=False)
        return
    #
    #
    def plot_network(self):
        # TODO
        if not self.network_prepped:
            self.prep_network()
        return





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
            print("Some of your input nodes didn't map to the reference databse ("+str(len(include_nodes_final))+" out of "+str(len(include_nodes))+" included).\nYou can see what's in the database by running diffuser.get_symbols()")
            #raise Warning
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
            print("Found some seeds that weren't in the reference ("+str(len(not_present_keys))+"). You can see what's in the database by running diffuser.get_symbols()")
            #raise Warning
        if len(present_keys)==0:
            raise Warning("Didn't find ANY of your seed values in our reference.\nYou can see what's in the database by running diffuser.get_symbols()")
        final_seed_value_dict = {k:seed_value_dict[k] for k in present_keys}
        return final_seed_value_dict
    #
    #
    def get_initial_seed_tensor(self,
                                include_idxs,
                                seed_value_dict,
                                include_symbols,
                                degree_vect):
        print(degree_vect)
        include_hash = {v:k for k, v in enumerate(include_symbols)}
        seed_tensor = torch.zeros((len(include_idxs)))
        # plug in the real values where the by belong
        for symbol, val in seed_value_dict.items():
            seed_tensor[include_hash[symbol]] = val
        seed_tensor /= degree_vect+1 # plus 1 to allow for orphan nodes otherwise, we end up with nans
        return seed_tensor
    #
    #
    def prep_diffuser(self,
                      seed_value_dict,
                      include_nodes = None,
                      degree_norm = True):
        print("preparing the primary diffuser")
        # run the real run
        include_idxs, include_symbols = self.get_final_indices(include_nodes)
        # filter just to make sure that there was correct mapping
        seed_value_dict = {k:v for k, v in seed_value_dict.items() if k in include_symbols}
        seed_value_dict = self.get_final_seed_value_dict(seed_value_dict)
        # Subset the interaction matrix for only the ones that are pertinent
        used_interaction_mat = self.interaction_mat[include_idxs,:]
        used_interaction_mat = used_interaction_mat[:,include_idxs]
        diffusion_matrix = torch.zeros((len(include_idxs),len(include_idxs)))
        diffusion_matrix[:,:]+=used_interaction_mat.astype(np.float32).todense()
        # Fill with the the degree if degree norm
        if degree_norm:
            # Calculate the degree of each node on the subnetwork that's included
            degree_vect = np.sum(used_interaction_mat,axis=0).flatten()
            idxs = torch.arange(len(include_idxs))
            diffusion_matrix[idxs,idxs]=torch.tensor(degree_vect.astype(np.float32).flatten())
        else:
            diffusion_matrix.fill_diagonal_(1.)
        ## Get the seed tensor
        seed_tensor = self.get_initial_seed_tensor(
                                include_idxs,
                                seed_value_dict,
                                include_symbols,
                                degree_vect)
        diffused = single_diffuser(include_symbols,
                                    used_interaction_mat,
                                    seed_tensor)
        return diffused



def make_stringdb_diffuser():
    symbols, interactions = read_dict('C:\\Users\\styler\\bin\\ppi_diffusion\\ppi_diffusion\\symbol_interaction_lilmat.pkl')
    return(diffuser(symbols, interactions))




