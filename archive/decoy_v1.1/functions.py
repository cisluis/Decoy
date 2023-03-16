import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplc

# Global Functions ################################################################

def animate(frames, nframes, cmap, norm, moviename, fps = 30):
        
    import matplotlib.animation as animation

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(8,8) )
    
    snapshots = [ frames[i, :, :] for i in range(nframes - 1) ]

    im = plt.imshow(snapshots[0], interpolation='nearest', aspect='auto', cmap=cmap, norm=norm)

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        im.set_array(snapshots[i])
        return [im]

    anim = animation.FuncAnimation(fig, animate_func, 
                                   frames = nframes, interval = 1000 / fps) 
    #plt.show()
    #anim.save( moviename, writer=animation.FFMpegWriter(fps=fps))
    #anim.save(moviename, fps=fps, extra_args=['-vcodec', 'libx264'])
    
    return(anim)
                     

# Parameter Functions ################################################################

class Params():
    """
    class of simulation parameters
    
    """
    def __init__(self, i = 0, filein = None):
        """
        creates set of parameters with default values
    
        """
        
        # Simulation 
        self.id = i                              # ID of the parameter set
        self.width = 100                         # Width of array grid
        self.height = 100                        # Height of array grid
        self.initial_population_0 = 100          # Initial population of SNVs
        self.initial_population_1 = 50           # Initial population of SHVs
        self.initial_population_2 = 10           # Initial population of RNVs
        self.number_of_steps = 100               # Time length of simulation (in ticks)
        self.tumor_radius = 10                   # Radius of initial tumor  
        
        # s controls
        self.drug_dossage = 100                  # Drug dosage per site  
        self.drug_dossage_time = 1000            # Time period for new dosage (+/- 1%)

        # Cell dynamics 
        self.life_span = 100                     # Age of cells natural death (+/- 10%)
        self.division_age = 5                    # Base age of cell division (+/- 5%)
        self.fitness_cost_0 = 1                  # Fitness factor for SNVs (multiply division age)
        self.fitness_cost_1 = 1.5                # Fitness factor for SHVs (multiply division age)
        self.fitness_cost_2 = 2                  # Fitness factor for RNVs (multiply division age)
        self.replacement_prob = 0.1              # Probability of cell replacement  
        self.kill_cell = 5                       # Killing threshold of cell by the drug
        
        # Vesicle dynamics
        self.mv_prod_0 = 0                       # Background vesicle production per cell per tick
        self.mv_prod_drug = 1                    # Vesicle production per cell per tick per unit drug
        self.mv_prod_max = 10                    # Maximum vesicle production per cell per tick
        self.kill_mv = 0.1                       # Max drug per MV (MV is `killed` when saturated)
        self.diff_mv = 0.1                       # Rate of MV diffusion (per tick per site)
        self.mv_max = 50                         # Maximum MVs in grid site (MV saturation)
        
        # Drug dynamics
        self.drug_abs_cell = 0.5                 # Rate of drug absorption per cell per tick
        self.drug_abs_mv = 0.1                   # Rate of drug absorption per vesicle per tick
        self.drug_decay = 0.1                    # Rate of drug decay per tick    
        self.diff_drug = 0.25                    # Rate of drug diffusion (per tick per site)    
        self.drug_max = 50                       # Maximum drug in grid site (drug saturation)
        
        # if a parameter filename is given, read parameters from it (index i)
        if filein is not None:
            self.read_parameters_from_file(filein, i)
        

    def get_parameter(self, param):

        return(getattr(self, param))

    def set_parameter(self, param, value):

        setattr(self, param, value)

    def read_parameters_from_file(self, fil, i):
        
        parameters = vars(self)
        
        tbl = pd.read_csv(fil)
        
        defaulted = [p for p in parameters if p not in tbl]
        extra = [p for p in tbl if p not in parameters]
        
        if defaulted:
            print("Attention! parameters: <" + str(defaulted) + \
              "> were not found in parameter table. They will be assigned default values...")
        if extra:
            print("Attention! parameters: <" + str(extra) + \
              "> in parameter table are not valid. They will be ignored...")
        
        for col in tbl:
            if col in parameters:
                self.set_parameter(col, tbl[col].values[i])
        
        
    def read_parameters_from_tbl(self, tbl, i):
        
        parameters = vars(self)
        
        defaulted = [p for p in parameters if p not in tbl]
        extra = [p for p in tbl if p not in parameters]
        
        if defaulted:
            print("Attention! parameters: <" + str(defaulted) + \
              "> were not found in parameter table. They will be assigned default values...")
        if extra:
            print("Attention! parameters: <" + str(extra) + \
              "> in parameter table are not valid. They will be ignored...")
        
        for col in tbl:
            if col in parameters:
                self.set_parameter(col, tbl[col].values[i])
                
