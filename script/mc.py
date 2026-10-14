import os
from ase.io import read, write
from ase import Atoms
import numpy as np
import sys
import  matplotlib 
import  matplotlib.pyplot as plt

import clusterx as c
from clusterx.utils import list_integer_named_folders, atat_to_cell, decorate_supercell
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.parent_lattice import ParentLattice
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.super_cell import SuperCell
from clusterx.visualization import plot_property_vs_concentration, plot_predictions_vs_target
from clusterx.visualization import plot_optimization_vs_number_of_clusters, plot_optimization_vs_sparsity
from clusterx.clusters_selector import ClustersSelector
from clusterx.structure import Structure
from clusterx.model import ModelBuilder
from clusterx.model import Model
"""
if 1:
    plat = ParentLattice(json_db_filepath="plat.json")
    plat.get_sublattice_types(pretty_print=True)
else:
    plat = atat_to_cell(file_path="lat.in", interpret_as="parent_lattice",pbc=(1,1,1))
    plat.serialize(fname="plat.json")
"""
if 0: # Build supercell from scratch
    scell = SuperCell(plat, [(4,0,0),(0,4,0),(0,0,1)])
    scell.get_sublattice_types(pretty_print=True)
    scell.serialize("scell.json")
else:
    scell = SuperCell(json_db_filepath = "scell.json")
    #scell.get_sublattice_types(pretty_print=True)
    print(scell)
        
if 0: # Build sset from a list of numbered folders containing structure files
	
    fl = list_integer_named_folders(prefix='', containing_files=["energy.dat", "geom.json"])
    sset = StructuresSet(plat)
        
    for path in fl:
        print(path)
        atoms = read(filename=os.path.join(str(path),"geom.json"))
        struc = decorate_supercell(scell, atoms)
        sset.add_structure(struc, folder=path)
		
    sset.serialize("sset.json")
    
elif 0: # Parse sset.json file
    sset = StructuresSet(json_db_filepath = "sset.json")
    
elif 0: # Exclude problematic structures
    sset0 = StructuresSet(json_db_filepath = "sset.json")
    str_idxs = []
    exclude = [42, 176]
    for i in range(len(sset0)):
        if i in exclude:
            continue
        else:
            str_idxs.append(i)
    sset = sset0.get_subset(str_idxs)
    
    
if 0: # Read properties from files and add to structures set object
    sset.set_property_values_from_files(property_name = "total_energy_centered", property_file_name = "energy.dat")

    
if 0: # Read properties from files and add to structures set object
    sset.set_property_values_from_files(property_name = "total_energy_fhi", property_file_name = "total-energy-fhi.dat")

    
if 0:
    #plot_property_vs_concentration(sset, property_name="energy")
    plot_property_vs_concentration(sset, property_name="total_energy_centered")
    

if 0: # Create cpool from scratch
    from clusterx.clusters.clusters_pool import ClustersPool
    cpool = ClustersPool(plat, npoints=[0,1,2,3,4], radii=[0,0,5,4.5,4.0], super_cell=scell)
    cpool.display_info()
    print('No. of clusters', len(cpool))
    cpool.serialize("cpool-4p.json")
    
elif 0:
    cpool = ClustersPool(json_db_filepath = "cpool_15.json")
    

#############

if 0:
    mb = ModelBuilder(basis="binary-linear",selector_type="identity",selector_opts={},estimator_type="skl_Lasso",estimator_opts={"fit_intercept":False, "alpha":0.000294})

    """
    mb = ModelBuilder(basis = "binary-linear",
                      selector_type="linreg",
                      selector_opts={'clusters_sets':'size'},
                      estimator_type="skl_LinearRegression",
                      estimator_opts={"fit_intercept":False})
    
    mb = ModelBuilder(basis = "binary-linear",
                      selector_type="lasso",
                      selector_opts={'sparsity_max': 0.1,'sparsity_min': 0.000001},
                      estimator_type="skl_RidgeCV",
                      estimator_opts={"fit_intercept":False,"alphas":np.logspace(-15,-1,500)})
    """
    cemodel = mb.build(sset, cpool, "total_energy_centered") #Build CE model using the training data set

    cpool_opt = mb.get_opt_cpool()
    cpool_opt.display_info(ecis=cemodel.get_ecis())
    cpool_opt.write_clusters_db(db_name="cpool_opt.json")
    
    cemodel.report_errors(sset)
    #plot_optimization_vs_number_of_clusters(mb.get_selector(),scale=0.7)
    #plot_optimization_vs_sparsity(mb.get_selector())
    
    cemodel.serialize("MODELBDR.pickle")
    
elif 1:
    cemodel = Model(filepath = "MODELBDR.pickle")

if 0:  # Select clusters according to the eci values.
    ecis=cemodel.get_ecis()
    ecis_n=[]
    select_index=[]
    for i in range(len(ecis)):
        if abs(ecis[i]) > 0.4990:
            ecis_n.append(ecis[i])
            select_index.append(i)
    print('no. of new ECIs', len(ecis_n),len(select_index))

    select_index=np.array(select_index)
    cpool_small=cpool_opt.get_subpool(select_index)
    cpool_small.display_info()

    mb2 = ModelBuilder(basis="binary-linear", selector_type="identity",selector_opts={},estimator_type="skl_Lasso",estimator_opts={"fit_intercept":False, "alpha":0.000294})
    cemodel2 = mb2.build(sset, cpool_small, "total_energy_centered")
    cpool_opt2 = mb2.get_opt_cpool()
    cpool_opt2.display_info(ecis=cemodel2.get_ecis())
    cpool_opt2.write_clusters_db(db_name="cpool_opt2.json")

    cemodel2.report_errors(sset)
    #plot_optimization_vs_number_of_clusters(mb.get_selector(),scale=0.7)
    #plot_optimization_vs_sparsity(mb.get_selector())

    cemodel2.serialize("cemodel-4p2.json")

if 0:
    plot_predictions_vs_target(sset, cemodel, "total_energy_centered")

if 0:
    plot_optimization_vs_number_of_clusters(mb.get_selector())
    

if 0:
    plot_property_vs_concentration(sset, cemodel = cemodel, property_name = "energy", show_loo_predictions=False)
    
if 0:
    errs = cemodel.get_all_errors(sset, srtd = True)
    errs1 = cemodel.get_cv_score(sset)
    #errs = cemodel.get_all_errors(sset, srtd = False)
    print(errs)
    print(errs1)
    
if 1: # MC for single temperature
    from clusterx.thermodynamics.wang_landau import WangLandau
   
    ensemble = "canonical"

    nsubs = {0:[6],2:[12]}
    emin = 8
    emax = 20

   
    wl = WangLandau(cemodel, scell = scell, nsubs = nsubs, ensemble = "canonical", predict_swap = True, error_reset=None)
    cods = wl.wang_landau_sampling(energy_range = [emin, emax], energy_bin_width = 0.01, f_range = [np.exp(1), np.exp(1e-7)],flatness_conditions=[[0.5, np.exp(1e-1)], [0.80, np.exp(1e-3)], [0.90,np.exp(1e-5)], [0.95,np.exp(1e-7)],[0.98,np.exp(1e-8)]], serialize = True, serialize_during_sampling = True, filename = 'cdos.json')

    nsites = len(scell.get_substitutional_atoms())
    print(nsites)

    temps = range(10,1500,10)
    scale_factor = [1]

    # Reading of configurational density of states from file, e.g. cdos2.json as produced before
    from clusterx.thermodynamics.wang_landau import ConfigurationalDensityOfStates
    cdos_read = ConfigurationalDensityOfStates(filename = 'cdos.json', read = True)

    e, lng = cdos_read.get_cdos(ln = True, normalization = True, normalization_type=1)
    from clusterx.visualization import plot_property
    plot_property(e,lng, xaxis_label = "E [eV]", yaxis_label = "ln(g(E))", prop_name="lng")
    plt.savefig("cdos.png")

    cp = cdos_read.calculate_thermodynamic_property( temps, prop_name = "C_p")
    plot_property(temps,cp, xaxis_label = "Temperature [K]", yaxis_label = "Specific heat [eV/K]", prop_name = "C_p")
    plt.subplots_adjust(left=0.2)  # 增加左侧边距，为Y轴标签腾出更多空间
    plt.savefig("C_p.png")

    u = cdos_read.calculate_thermodynamic_property( temps, prop_name = "U")
    plot_property(temps,u, xaxis_label = "Temperature [K]", yaxis_label = "Internal energy [eV]", prop_name = "U")
    plt.savefig("U.png")

    f = cdos_read.calculate_thermodynamic_property( temps, prop_name = "F")
    plot_property(temps,f, xaxis_label = "Temperature [K]", yaxis_label = "Free energy [eV]", prop_name = "F")
    plt.savefig("F.png")

    s = cdos_read.calculate_thermodynamic_property( temps, prop_name = "S")
    plot_property(temps,s, xaxis_label = "Temperature [K]", yaxis_label = "entropy [eV/K]", prop_name = "S")
    plt.subplots_adjust(left=0.2)  # 增加左侧边距，为Y轴标签腾出更多空间
    plt.savefig("S.png")


def two_level_analytic(temp, e1, e2, kB = 8.6173303e-5):
    """Compute expected energy of a two level system at temperature ``temp``
    for a two level system with energies e1 and e2. 

    **Parameters:**
    ``temp``: float
        temperature
    
    ``e1``: float
        Energy level of the ground state
    
    ``e2``: float
        Energy level of the excited state

    ``kB``: float (default: 8.6173303e-5)
        Boltzmann constant in units of [e1]/[temp]. The default value corresponds
        to eV/Kelvin, so temperatures must be given in Kelvin and energies in eV
        if the default value is used.
    """
    de = e2 - e1

    beta = 1.0 / (kB * temp)
    e = e1 + de / (1 + np.exp(beta * de))
    return e

if 0: # MC for internal energy
    from clusterx.thermodynamics.monte_carlo import MonteCarlo
    from clusterx.thermodynamics.monte_carlo import MonteCarloTrajectory
    ensemble = "canonical"

    e1abinitio = -0.772
    e2abinitio = 1.635
    e1model = -0.8139703
    e2model = 1.42952218
    
    nmc = 1000
    neq = 200
    kb = 8.6173303e-5 # Boltzmann constant in eV/K
    nsubs = {0:[1]}
    temp1 = 40000
    temp2 = 3000
    dtemp = -4000
    temps = []
    internal_energy = []

    f = open("u_vs_t.dat", "w")
    
    for i, temp in enumerate(range(temp1, temp2, dtemp)):
        temps.append(temp)
        filena = "trajectory-T"+str(temp)+".json"
        mc = MonteCarlo(cemodel, scell, nsubs, filename = filena, predict_swap = False)
    
        traj = mc.metropolis(no_of_sampling_steps = nmc, temperature = temp, boltzmann_constant = kb, scale_factor = [1], write_to_file= True)

        internal_energy.append( traj.calculate_average_property(prop_name = "U", no_of_equilibration_steps = neq))

        umean_ana_abinitio = two_level_analytic(temp, e1abinitio, e2abinitio)
        umean_ana_model = two_level_analytic(temp, e1model, e2model)
        f.write(f'{temps[i]}\t{internal_energy[i]}\t{umean_ana_abinitio}\t{umean_ana_model}\n')
        f.flush()
        
        a = traj.get_energies()
        print(a)

        # traj.serialize()

    f.close()
        
