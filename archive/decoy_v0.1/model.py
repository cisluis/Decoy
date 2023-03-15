import math
import random
import numpy as np
import pandas as pd
import mesa

from agents import event_probability, Cell, MV, Drug

def euclid(pos_1, pos_2):
    """ Get the Euclidean distance between two points
        Args:
            pos_1, pos_2: Coordinate tuples
    """
    return math.sqrt((pos_1[0] - pos_2[0])**2 + (pos_1[1] - pos_2[1])**2)


def diskmask(shape, radius):
    """ Produces a circular mask in the center of a grid
        Args:
            shape: of array grid
            radius: radius of mask
    """
    from skimage.draw import disk
    
    mask = np.zeros(shape)    
    rr, cc = disk((shape[0]/2, shape[0]/2), radius)
    mask[rr, cc] = 1
    
    return mask

# model object ################################################################

class Decoy(mesa.Model):
    """
    Decoy Model
    """
    def __init__(self, params, verbose = False):
        """
        Create a new model 
        """

        # Set parameters
        self.params = params
        self.id = params.id
        self.width = params.width
        self.height = params.height
        self.time = 0
        self.last_dossage_time = 0
        self.verbose = verbose            # Print-monitoring
        self.dead_agents = list()         # List of dead agents per step
        self.moved_agents = list()        # List of moved agents per step
        self.born_agents = list()         # List of born agents per step
        
        
        # local variables 
        initial_population = list((params.initial_population_0,
                                   params.initial_population_1,
                                   params.initial_population_2))
                                  
        # division age
        self.div_ages = list((params.division_age * params.fitness_cost_0,
                              params.division_age * params.fitness_cost_1,
                              params.division_age * params.fitness_cost_2))
         
        
        # scheduler for simultaneous activation of all the agents
        self.schedule = mesa.time.SimultaneousActivation(self)
        # model grid allows for several agents per site
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=True)
        
        # NEEDS WORK HERE!!!!!
        # schedule data collection
        mr={"Populations": self.compute_populations}
        self.datacollector = mesa.DataCollector(model_reporters= mr)

        agent_id = 0
        
        # Create field agents: drug and mv
        for _, x, y in self.grid.coord_iter():
            drug = Drug(agent_id, self, (x, y), 0)
            agent_id += 1
            self.grid.place_agent(drug, (x, y))
            self.schedule.add(drug)
            mv = MV(agent_id, self, (x, y))
            agent_id += 1
            self.grid.place_agent(mv, (x, y))
            self.schedule.add(mv)
    
        # add drug to the system
        self.add_drug()
        
        # Create cell agents
        def select_in_mask(w, h, mask):
            # select random (x,y) unocuppied coordinates in mask
            x = self.random.randrange(w)
            y = self.random.randrange(h)
            while (mask[x,y] == 0 or self.occupied((x,y))):
                x = self.random.randrange(w)
                y = self.random.randrange(h)
            return (x,y)
        
        # tumor mask (M is its total size)
        #mask = np.ones((self.width, self.height))
        mask = diskmask((self.width, self.height), self.params.tumor_radius)
        M = np.sum(mask)
        
        # N is the total initial population
        N = np.sum(initial_population)
        for typ in (0, 1, 2):
            if (N > M):
                # if tumor size is too small, renormalizes populations to fit
                initial_population[typ] = (initial_population[typ]/N)*M
            # makes sure there is at least one cell of each kind
            initial_population[typ] = int(np.max([initial_population[typ], 1]))
            for i in range(initial_population[typ]):
                # find a random location
                x, y =  select_in_mask(self.width, self.height, mask)
                             
                # pick a random initial age within reproductive age
                age = np.random.randint(self.div_ages[typ], size=1)[0]
                cell = Cell(agent_id, self, (x, y), typ, age, 0)
                agent_id += 1
                self.grid.place_agent(cell, (x, y))
                self.schedule.add(cell)
        
        self.max_id = agent_id
        self.running = True
        self.datacollector.collect(self)
  

    def compute_populations(self):
        """
        Get population abundances for each phenotype
        """
        population = [0, 0, 0]
        for agent in self.schedule.agents:
            if type(agent) is Cell:
                population[agent.phenotype] += 1
        return population
    
    
    def cell_table(self):
        """
        Get table of all living cell properties
        """
        cell_props = {"time": [], "cell_id": [], "phenotype": [],
                      "age": [], "x": [], "y": [], "drug": []}
        for agent in self.schedule.agents:
            if type(agent) is Cell:
                if agent.alive:
                    cell_props["time"].append(self.time)
                    cell_props["cell_id"].append(agent.unique_id)
                    cell_props["phenotype"].append(agent.unique_id)
                    cell_props["age"].append(agent.age)
                    cell_props["x"].append(agent.pos[0])
                    cell_props["y"].append(agent.pos[1])
                    cell_props["drug"].append(agent.drug)
        return pd.DataFrame(cell_props)
    
    
    def get_agent(self, pos, typ):
        # get ALL agents of type typ in this location
        agents = self.get_agents(pos, typ)
        return agents[0]
    
    
    def get_agents(self, pos, typ):
        # get ALL agents of type typ in this location
        agents_out = list()
        agents = self.grid.get_cell_list_contents([pos])
        for agent in agents:
            if type(agent) is typ:
                agents_out.append(agent)
        return agents_out
        
        
    def occupied(self, pos):
        # get if 'pos' is occupied by a cell agent
        agents = self.grid.get_cell_list_contents([pos])
        return any(isinstance(agent, Cell) for agent in agents)
    
    
    def add_drug(self):
        #drug_distribution = np.genfromtxt("drug-map.txt")
        drug_distribution = self.params.drug_dossage*np.ones((self.width, self.height))
        for _, x, y in self.grid.coord_iter():
            dose = drug_distribution[x, y]
            drug = self.get_agent((x,y), Drug)
            #drug.amount = np.min([drug.amount + dose, self.params.drug_max])
            drug.amount = dose

    
    def step(self):
                     
        # Time step of the model 
        self.time += 1
        self.schedule.step()  # runs all scheduled events
        
        # updates states of agents 
        # (this has to be done here because the schedule is simultaneous)
        # NOTE: This might be done using the agent.advance attribute, but not sure how yet 
        
        # clean out agents scheduled to move which are also scheduled to die
        # (e.g. an agent is scheduled to die by replacement after it was scheduled to move)
        self.moved_agents = [agent for agent in self.moved_agents if agent not in self.dead_agents]
                
        # refresh dead agents in model schedule and grid
        self.dead_agents = list(set(self.dead_agents)) 
        for agent in self.dead_agents:
            #print("died: ID:" + str(agent.unique_id) + "; pos = " + str(agent.pos))
            self.schedule.remove(agent)
            self.grid.remove_agent(agent)
            #self.dead_agents.remove(agent)
        self.dead_agents = list()

        # refresh born agents in model schedule and grid
        self.born_agents = list(set(self.born_agents)) 
        for agent in self.born_agents:   
            if (len(self.get_agents(agent.pos, Cell)) == 1):
                # if new cell is the only one in 'pos', places and schedules it
                #print("Born, " + str(self.time) + " ID:" + str(agent.unique_id) + "; pos = " + str(agent.pos))
                self.grid.place_agent(agent, agent.pos)
                self.schedule.add(agent)
            else:
                # if another agent is already in place, abort cell birth
                #print("Aborted, " + str(self.time) + "; ID:" + str(agent.unique_id) + "; pos = " + str(agent.pos))
                self.grid.place_agent(agent, agent.pos)
                self.grid.remove_agent(agent)
            #self.born_agents.remove(agent)
        self.born_agents = list()
        
        # refresh moved agents in model grid
        self.moved_agents = list(set(self.moved_agents)) 
        for agent in self.moved_agents: 
            if self.occupied(agent._pos):
                # if position is already taken (eg. two agents moved to the same spot) kills agent
                #print("Killed, " + str(self.time) + "; ID:" + str(agent.unique_id) + "; pos = " + str(agent.pos)+ "; _pos = " + str(agent._pos))
                self.schedule.remove(agent)
                self.grid.remove_agent(agent)
            else: 
                # move agent to the grid position assigned in agent.move()
                #print("Moved, " + str(self.time) + "; ID:" + str(agent.unique_id) + "; pos = " + str(agent.pos)+ "; _pos = " + str(agent._pos))
                self.grid.move_agent(agent, agent._pos)
            #self.moved_agents.remove(agent)
        self.moved_agents = list()
        
        # adds more drug to the system
        dossage_time = self.time - self.last_dossage_time
        if (np.random.random(1)[0] < event_probability(dossage_time, 
                                                       self.params.drug_dossage_time, 
                                                       self.params.drug_dossage_time/100)):
            self.last_dossage_time = self.time
            self.add_drug()
           
        # collect data
        self.datacollector.collect(self)
        if self.verbose:
            print([self.schedule.time, self.compute_populations()], end="\r")

            
    def get_arrays(self):
        
        cellgrid = np.zeros((self.width, self.height))
        mvgrid = np.zeros((self.width, self.height))
        druggrid = np.zeros((self.width, self.height))
    
        for pos in self.grid.coord_iter():
            agents, x, y = pos

            cell = [agent.phenotype for agent in agents if type(agent) is Cell]
            if (len(cell)==0):
                cell = [-1]
            mv = [agent.amount for agent in agents if type(agent) is MV]
            drug = [agent.amount for agent in agents if type(agent) is Drug]

            cellgrid[y, x] = cell[0]
            mvgrid[y, x] = mv[0]
            druggrid[y, x] = drug[0]
        
        return (cellgrid, mvgrid, druggrid)
    
    
    def collect_data(self, i, cell_grid, mv_grid, drug_grid):
        
        cellgrid, mvgrid, druggrid = self.get_arrays()
        cell_grid[i, :, :] = cellgrid
        mv_grid[i, :, :] = mvgrid
        drug_grid[i, :, :] = druggrid 
       
        return (cell_grid, mv_grid, drug_grid)
        
        
    def run_model(self, fileout):
        
        steps = self.params.number_of_steps
        
        premature_finish = False
        
        cell_grid = np.zeros((steps, self.width, self.height))
        mv_grid = np.zeros((steps, self.width, self.height))
        drug_grid = np.zeros((steps, self.width, self.height))
        t_series = pd.DataFrame({"Step":[],"SNVs":[],"SHVs":[],"RNVs":[]})
        
        N0, N1, N2 = list(self.compute_populations())
        t_series = pd.concat([t_series, pd.DataFrame({"Step" : [0], 
                                                      "SNVs": [N0], 
                                                      "SHVs": [N1], 
                                                      "RNVs": [N2]})])
        cell_table = self.cell_table()
        cell_grid, mv_grid, drug_grid = self.collect_data(0, cell_grid, mv_grid, drug_grid)
        
        if self.verbose:
            print( "Initial Population: %s" % ((N0, N1, N2),))

        for i in range(steps - 1):
            
            self.step()
            
            N0, N1, N2 = list(self.compute_populations())
            
            if (np.sum([N0, N1, N2]) <= 0):
                print('Whole population died!!!')
                premature_finish = True
                break
                
            if (np.sum([N0, N1, N2]) > self.width*self.height*0.8):
                print('Population grew too big!!!')
                premature_finish = True
                break
            
            t_series = pd.concat([t_series, pd.DataFrame({"Step" : [i + 1],
                                                          "SNVs": [N0],
                                                          "SHVs": [N1],
                                                          "RNVs": [N2]})])
            cell_table = pd.concat([cell_table, self.cell_table()]) 
            cell_grid, mv_grid, drug_grid = self.collect_data(i + 1, cell_grid, mv_grid, drug_grid)
        
        
        if self.verbose:
            print("#################")
            print("Final Population: %s" % ((N0, N1, N2),))
            
            
        # reduces outputs to actual final 
        if premature_finish:
            cell_grid = cell_grid[0:i, :, :]
            mv_grid = mv_grid[0:i, :, :]
            drug_grid = drug_grid[0:i, :, :]
        
        # saves output into a file
        np.savez_compressed(fileout,
                            cell_grid=cell_grid,
                            mv_grid=mv_grid,
                            drug_grid=drug_grid,
                            t_series=t_series.to_numpy(),
                            cell_table=cell_table.to_numpy())
            
        return (cell_grid, mv_grid, drug_grid, t_series, cell_table)    
            