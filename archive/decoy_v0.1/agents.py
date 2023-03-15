import math
import random
import numpy as np
import pandas as pd
import mesa


def event_probability(t, t_mean, t_std):
    """ Get the probability of an event from a cdf distribution with mean, std mean
        Args:
            t:      time variable value (eg. age)
            t_mean: mean time of event (eg. life span) 
        
    """
    from scipy.stats import norm
    return norm.cdf(t, loc=t_mean, scale=t_std)


def sigmoid(x, mu, K):
    """ Response function: 
        Args:
            mu: response is linear with slope 'mu' for x small (x->0) 
            K: response saturates at value K for x large (x->inf)  
    """
    return K*(mu*x / (mu*x + K))


def laplacian9p(agent, typ):
    """
    Discrete Laplace operator using a nine-point stencil (kernel):

          | 0.25 0.50 0.25 |
    D^2 = | 0.50 -3.0 0.50 | *(1/3)
          | 0.25 0.50 0.25 |

    We can define a factor = (0.5/(|x - x_0| + |y - y_0|) which is 
     - 0.25 for diagonal elements
     - 0.50 for non-diagonal elements
     - undefined for center element (irrelevant)

    """

    # Calculate discrete Laplacian in Moore's neighborhood
    norm = 1/3
    dif = -3*norm*agent.amount

    # Good way but it doesn't work well in the edges of the torus
    #neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
    #for neighbor in neighbors:
    #    if type(neighbor) is type(agent):        
    #        factor = 1/(np.abs(neighbor.pos[0] - self.pos[0]) + np.abs(neighbor.pos[1] - self.pos[1]))
    #        dif += 0.5*norm*factor*neighbor.amount

    neighbors = [(-1, -1),  (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dpos in neighbors:
        den = np.abs(dpos[0]) + np.abs(dpos[1])
        pos = ((agent.pos[0] + dpos[0])% agent.model.width, 
               (agent.pos[1] + dpos[1])% agent.model.height)
        val = agent.model.get_agent(pos, typ).amount
        dif += 0.5*norm*val/den

    return dif


# Agent objects ################################################################


## Cell Agent class
class Cell(mesa.Agent):
    """
    Cell agent
    
    """
    def __init__(self, unique_id, model, pos, phenotype, age = 0, drug=0):
        """
        Cell agent constructor:
        
        (*) pos = (x,y) position in grid
        (*) phenotype = 0: SNV: sensitive cell (wild type with normal-vesicularization) 
                        1: SHV: sensitive cell with hightened-vesicularization 
                                (produces vesicles in response to drug stress) 
                        2: RNV: drug resistant cell (normal-vesicularization)

        (*) drug = amount of drug toxicity, if (drug > MU_DRUG) bug dies
        age = age in ticks

        """
        super().__init__(unique_id, model)
        self.pos = pos
        self._pos = pos
        self.phenotype = phenotype
        self.drug = drug
        self.age = age
        self.alive = True
        return    

    
    def move(self):
        # move cell to a neighboring grid site
        moved = False
        neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        candidates = [ pos for pos in neighborhood if not self.model.occupied(pos)]
        
        if candidates:
            # if available locations pick random one
            new_pos = self.random.choice(candidates)
            self._pos = new_pos
            self.model.moved_agents.append(self)
            moved = True
        elif (np.random.random(1)[0] < self.model.params.replacement_prob):
            # if there aren't available locations pick a random neighbor to replace
            new_pos = self.random.choice(neighborhood)
            dead_agent = self.model.get_agent(new_pos, Cell)
            dead_agent.die()
            #print("Death by replacement: ID=" + str(dead_agent.unique_id))
            self._pos = new_pos
            self.model.moved_agents.append(self)
            moved = True
            
        return moved
    
            
    def life_cycle(self):
        
        if (self.alive):
            rep_age = self.model.div_ages[self.phenotype]
            life_span = self.model.params.life_span

            if (np.random.random(1)[0] < event_probability(self.age, life_span, life_span/10)):
                # random natural death for the bug's age
                # print("Death by old age: ID=" + str(self.unique_id) + '; age=' + str(self.age))
                self.die()
            elif (np.random.random(1)[0] < event_probability(self.age, rep_age, rep_age/20)):
                # random replication for the cell's age; try to move out of pos and divide
                pos = self.pos
                if (self.move()):
                    # if it can move, then divides an place daughter in original pos
                    # print("Cell division: ID=" + str(self.unique_id) + '; age=' + str(self.age))
                    self.age = 0
                    self.drug = self.drug/2   # divide drug between childs
                    self.divide(pos, self.phenotype, self.drug)
        return

    
    def drug_in(self):
        # cell takes drug (if phenotype is SNV or SHV)
        Ac = self.model.params.drug_abs_cell
        
        if (self.phenotype < 2 and self.alive):
            # find amount of drug in cell location
            drug_patch = self.model.get_agent(self.pos, Drug)
            # dose is capped at 'Ac'
            dose = np.min([drug_patch.amount, Ac])
            # cell takes the drug
            self.drug += dose
            drug_patch.amount -= dose
            if self.drug > self.model.params.kill_cell:
                # print("Death by drug: ID=" + str(self.unique_id))
                self.die()
    
    
    def vesiculate(self):
        # cell vesiculates
        NU_0 = self.model.params.mv_prod_0
        NU_D = self.model.params.mv_prod_drug
        NU_MAX = self.model.params.mv_prod_max
        MAX_MVS_SITE = self.model.params.mv_max
        
        mv_patch = self.model.get_agent(self.pos, MV)
        # if cell is SHV
        if (self.phenotype == 1):
            # finds amount of drug in location
            drug_patch = self.model.get_agent(self.pos, Drug)
            # Vesicle production is a response to drug amount in location
            mvs = NU_0 + sigmoid(drug_patch.amount, NU_D, NU_MAX) 
        else:
            # background vesicle production
            mvs = NU_0
        # updates amount of vesicles in site, capped by 'MAX_MVS_SITE'
        mv_patch.amount = np.min([mv_patch.amount + mvs, MAX_MVS_SITE])
        # print("Vesiculation: " + str(mv_patch.amount) + " at: " + str(mv_patch.pos))
    
    
    def divide(self, pos, phenotype, drug):
        #print("Cell divide: ID=" + str(self.unique_id) + " at:" + str(self.pos))
        self.model.max_id += 1
        new_cell = Cell(self.model.max_id, self.model, pos, phenotype, drug)
        #print("Daughter: ID=" + str(new_cell.unique_id) + " at:" + str(new_cell.pos))
        self.model.born_agents.append(new_cell)
            

    def die(self):
        #print("Cell died: ID=" + str(self.unique_id) + " at:" + str(self.pos))
        self.alive = False
        if self not in self.model.dead_agents:
            self.model.dead_agents.append(self)
    
    
    def step(self):
        self.drug_in()
        self.life_cycle()

        
    def advance(self):
        if self.alive:
            self.vesiculate()
            self.age += 1
        
        
######################################################################################################

class MV(mesa.Agent):
    """
    Vesicle patch agent: it exist on each position and the 'amount' accounts for local concentration
    
    """
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.amount = 0.0
        self.drug = 0.0
        self._nextAmount = 0.0
        
        
    def drug_in(self):
        # absorbtion of drug by MVs
        Av = self.model.params.drug_abs_mv
        MAX_DRUG = self.model.params.kill_mv
        
        if (self.amount > 0):
            drug_patch = self.model.get_agent(self.pos, Drug)
            # dose is capped at 'Av' per tick per vesicle
            dose = np.min([drug_patch.amount, Av*self.amount])
            if (dose > 0):
                # get number of vesicles that would saturate with this amount of drug 
                # leftover is the residual drug after saturating MVs
                num_mvs, leftover = divmod(self.drug + dose, self.amount*MAX_DRUG) 

                # if all vesicles in location saturate
                if (num_mvs > self.amount):
                    # the amount of drug to cap out all MVs
                    dose = self.amount*MAX_DRUG - self.drug
                    # and no MVs are left out
                    num_mvs = self.amount
                    leftover = 0
                # update amounts
                self.amount -= num_mvs
                self.drug = leftover
                drug_patch.amount -= dose
 

    def diffuse(self):
        # Calculate amount of change due to difussion
        return self.model.params.diff_mv * laplacian9p(self, MV)

    def step(self):
        # vesicle difussion using discrete Laplacian 
        self._nextAmount = self.amount + self.diffuse()
        
        
    def advance(self):
        """
        Set the state to the new computed state -- computed in step().
        """
        self.amount = self._nextAmount
        #print("MV: " + str(self.amount) + " at: " + str(self.pos))
        self.drug_in()


#####################################################################################################

class Drug(mesa.Agent):
    """
    Drug patch agent: it exist on each position and the 'amount' accounts for local concentration
    
    """
    def __init__(self, unique_id, model, pos, drug):
        super().__init__(unique_id, model)
        self.pos = pos
        self.amount = drug
        self._nextAmount = 0.0

    
    def diffuse(self):
        # Calculate amount of change due to diffusion
        return self.model.params.diff_drug * laplacian9p(self, Drug)
    
    def step(self):
        # drug difussion using discrete Laplacian 
        self._nextAmount = self.amount + self.diffuse()
       
    
    def advance(self):
        """
        Set the state to the new computed state -- computed in step().
        """
        self.amount = self._nextAmount*math.exp(-self.model.params.drug_decay)
        