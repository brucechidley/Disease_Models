import numpy as np
from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass

import numba
from numba import int32, int64
from numba.experimental import jitclass

from repast4py import core, space, schedule, logging, random
from repast4py import context as ctx
from repast4py.parameters import create_args_parser, init_params

from repast4py.space import ContinuousPoint as cpt
from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType

import Kingston_Info_Comb

@dataclass
class MeetLog:
    susceptible_count: int = 0
    exposed_count: int = 0
    infectious_count: int = 0
    recovered_count: int = 0
    total_count: int = 0
    masked_count: int = 0
    vaccinated_count: int = 0


class Person(core.Agent):

    TYPE = 0

    def __init__(self, local_id: int, rank: int, agent_info, params_sa):
        super().__init__(id=local_id, type=Person.TYPE, rank=rank)

        self.info = agent_info

        self.real_coords = [self.info[1], self.info[2], self.info[3], self.info[4]]

        self.home_loc = [self.info[1][0] * (-100000), self.info[1][1] * 100000]
        self.job_loc = [self.info[2][0] * (-100000), self.info[2][1]  * 100000]
        self.store_1_loc = [self.info[3][0] * (-100000), self.info[3][1]  * 100000]
        self.store_2_loc = [self.info[4][0] * (-100000), self.info[4][1]  * 100000]


        self.age_bracket = agent_info[0][2]
        self.isolating = False

        vaccine_factor = params_sa[2]
        mask_chance = params_sa[3]
        mask_factor = params_sa[4]

        #Assigns them as infectious from the start depending on some probability (10% here)
        prob_infectious = np.random.uniform(0,1)
        if (prob_infectious <= 0.1):
            self.susceptible = False
            self.exposed = False
            self.infectious = True
            self.recovered = False
            self.total_time_in_class = np.random.normal(16, 4)
        
        else:
            self.susceptible = True
            self.exposed = False
            self.infectious = False
            self.recovered = False
            self.total_time_in_class = 999

        #Same deal with masking and vaccinating
        prob_masked = np.random.uniform(0,1)
        if (prob_masked <= mask_chance):
            self.masked_factor = mask_factor
        else:
            self.masked_factor = 1

        prob_vaccinated = np.random.uniform(0,1)

        #Set to 1 by default
        self.vaccinated_factor = 1

        #Vaccination rates based on age bracket
        if self.age_bracket == 0:
            if (prob_vaccinated <= 0.251):
                self.vaccinated_factor = vaccine_factor
        elif self.age_bracket == 1:
            if (prob_vaccinated <= 0.771):
                self.vaccinated_factor = vaccine_factor
        elif self.age_bracket == 2:
            if (prob_vaccinated <= 0.819):
                self.vaccinated_factor = vaccine_factor
        elif self.age_bracket == 3:
            if (prob_vaccinated <= 0.851):
                self.vaccinated_factor = vaccine_factor
        elif self.age_bracket == 4:
            if (prob_vaccinated <= 0.883):
                self.vaccinated_factor = vaccine_factor
        elif self.age_bracket == 5:
            if (prob_vaccinated <= 0.885):
                self.vaccinated_factor = vaccine_factor
        elif self.age_bracket == 6:
            if (prob_vaccinated <= 0.940):
                self.vaccinated_factor = vaccine_factor
        elif self.age_bracket == 7:
            if (prob_vaccinated <= 0.982):
                self.vaccinated_factor = vaccine_factor
        elif self.age_bracket == 8:
            if (prob_vaccinated <= 0.990):
                self.vaccinated_factor = vaccine_factor

        self.time_in_class = 0

    def move(self):

        if model.time_of_day == 1:
            model.space.move(self, cpt(self.home_loc[0], self.home_loc[1]))
        
        else:

            if model.day_of_week <= 10:
                model.space.move(self, cpt(self.job_loc[0], self.job_loc[1]))
            elif model.day_of_week == 12:
                model.space.move(self, cpt(self.store_1_loc[0], self.store_1_loc[1]))
            else:
                model.space.move(self, cpt(self.store_2_loc[0], self.store_2_loc[1]))
                
    def count_states(self, meet_log: MeetLog):

        if self.susceptible:
            meet_log.susceptible_count += 1
        elif self.exposed:
            meet_log.exposed_count += 1
        elif self.infectious:
            meet_log.infectious_count += 1
        else:
            meet_log.recovered_count += 1

        meet_log.total_count += 1

        if self.masked_factor < 1:
            meet_log.masked_count += 1

        if self.vaccinated_factor < 1:
            meet_log.vaccinated_count += 1
                
    def save(self):
        return (self.uid,[self.info, self.real_coords, self.home_loc, self.job_loc, self.store_1_loc, self.store_2_loc, self.age_bracket, self.isolating,
                          self.susceptible, self.exposed, self.infectious, self.recovered, self.masked_factor, self.vaccinated_factor, self.time_in_class], [])
    
    
    def update(self, data: bool):
        """Updates the state of this agent when it is a ghost
        agent on some rank other than its local one.

        Args:
            data: the new agent state (received_rumor)
        """

        print("here")

        self.info = data[1][0]
        self.real_coords = data[1][1]
        self.home_loc = data[1][2]
        self.job_loc = data[1][3]
        self.store_1_loc = data[1][4]
        self.store_2_loc = data[1][5]
        self.age_bracket = data[1][6]
        self.isolating = data[1][7]
        self.susceptible = data[1][8]
        self.exposed = data[1][9]
        self.infectious = data[1][10]
        self.recovered = data[1][11]
        self.masked_factor = data[1][12]
        self.vaccinated_factor = data[1][13]
        self.time_in_class = data[1][14]
    
    

class Location(core.Agent):

    TYPE = 1

    def __init__(self, a_id, rank, loc_info):
        super().__init__(id=a_id, type=Location.TYPE, rank=rank)

        self.real_coords = [loc_info[3][0], loc_info[3][1]]
        self.adjusted_coords = [loc_info[3][0] * (-100000), loc_info[3][1] * 100000]

        self.all_info = loc_info

        self.loc_type = loc_info[1]

        self.agent_count = 0
        self.susceptible_count = 0
        self.exposed_count = 0
        self.infectious_count = 0
        self.recovered_count = 0
    
    def update_info(self):

        c_space = model.space

        all_agents_at_loc = list(c_space.get_agents(cpt(self.adjusted_coords[0],self.adjusted_coords[1])))

        try:
            all_agents_at_loc.remove(self)
        except:
            print("Continue")

        new_agent_list = []

        for item in all_agents_at_loc:
            if item.uid[1] == Location.TYPE:
                all_agents_at_loc.remove(item)
            else:
                if item.info[0][0] in self.all_info[2]:
                    new_agent_list.append(item)
    

        count = 0
        susceptibles = 0
        exposeds = 0
        infectiouses = 0
        recovereds = 0

        for agent_at_loc in new_agent_list:

            count += 1

            if agent_at_loc.susceptible:
                susceptibles += 1
            elif agent_at_loc.exposed:
                exposeds += 1
            elif agent_at_loc.infectious:
                infectiouses += 1
            else:
                recovereds += 1

        self.agent_count = count
        self.susceptible_count = susceptibles
        self.exposed_count = exposeds
        self.infectious_count = infectiouses
        self.recovered_count = recovereds

    def update_agent(self, current_agent, all_agents):

        newly_exposed = False

        if current_agent.susceptible:

            for agent_at_same_loc in all_agents:

                if ((agent_at_same_loc.infectious) and (not agent_at_same_loc.isolating)):

                    odds = (agent_at_same_loc.masked_factor * current_agent.masked_factor) * (current_agent.vaccinated_factor) * (model.basic_rep_ratio / 16) / (self.susceptible_count)

                    against = np.random.uniform(0,1)
                    if (against < odds):
                        newly_exposed = True


            if newly_exposed:
                current_agent.susceptible = False
                current_agent.exposed = True
                current_agent.time_in_class = 0

                #Set the time the agent will be in the exposed class
                current_agent.total_time_in_class = np.random.normal(9, 2)

        elif current_agent.exposed and current_agent.time_in_class >= current_agent.total_time_in_class:
            current_agent.exposed = False
            current_agent.infectious = True
            current_agent.time_in_class = 0

            #Set the time the agent will be in the infectious class
            current_agent.total_time_in_class = np.random.normal(16, 4)

            #Check to see if agent isolates
            isolating_prob = np.random.uniform(0, 1)
            if isolating_prob <= model.isolation_rate:
                current_agent.isolating = True

        #Same for infectious
        elif current_agent.infectious and current_agent.time_in_class >= current_agent.total_time_in_class:
            current_agent.infectious = False
            current_agent.recovered = True
            current_agent.time_in_class = 0

            #Set the time the agent will be in the recovered class
            current_agent.total_time_in_class = 14

            current_agent.isolating = False
        
        #Same for recovered
        elif current_agent.recovered and current_agent.time_in_class >= current_agent.total_time_in_class:
            current_agent.recovered = False
            current_agent.susceptible = True
            current_agent.time_in_class = 0

            #Set to 999, since an agent is susceptible until infected
            current_agent.total_time_in_class = 999


        current_agent.time_in_class += 1

    def step(self):

        c_space = model.space

        all_agents_at_loc = list(c_space.get_agents(cpt(self.adjusted_coords[0],self.adjusted_coords[1])))

        try:
            all_agents_at_loc.remove(self)
        except:
            print("Continue")

        new_agent_list = []

        for item in all_agents_at_loc:
            if item.uid[1] == Location.TYPE:
                all_agents_at_loc.remove(item)
            else:
                if item.info[0][0] in self.all_info[2]:
                    new_agent_list.append(item)

        for agent_at_loc in new_agent_list:

            self.update_agent(agent_at_loc, new_agent_list)

    def save(self):

        return (self.uid,[], [self.all_info, self.real_coords, self.adjusted_coords, self.loc_type, self.agent_count, self.susceptible_count, self.exposed_count, self.infectious_count, self.recovered_count])

agent_cache = {}

def restore_agent(agent_data: Tuple):
    """Creates an agent from the specified agent_data.

    This is used to re-create agents when they have moved from one MPI rank to another.
    The tuple returned by the agent's save() method is moved between ranks, and restore_agent
    is called for each tuple in order to create the agent on that rank. Here we also use
    a cache to cache any agents already created on this rank, and only update their state
    rather than creating from scratch.

    Args:
        agent_data: the data to create the agent from. This is the tuple returned from the agent's save() method
                    where the first element is the agent id tuple, and any remaining arguments encapsulate
                    agent state.
    """

    uid = agent_data[0]
    # 0 is id, 1 is type, 2 is rank

    if uid[1] == Person.TYPE:
        if uid in agent_cache:
            agent_re = agent_cache[uid]
        else:
            agent_re = Person(uid[0], uid[2], agent_data[1][0])
            agent_cache[uid] = agent_re

        # restore the agent state from the agent_data tuple
        agent_re.info = agent_data[1][0]
        agent_re.real_coords = agent_data[1][1]
        agent_re.home_loc = agent_data[1][2]
        agent_re.job_loc = agent_data[1][3]
        agent_re.store_1_loc = agent_data[1][4]
        agent_re.store_2_loc = agent_data[1][5]
        agent_re.age_bracket = agent_data[1][6]
        agent_re.isolating = agent_data[1][7]
        agent_re.susceptible = agent_data[1][8]
        agent_re.exposed = agent_data[1][9]
        agent_re.infectious = agent_data[1][10]
        agent_re.recovered = agent_data[1][11]
        agent_re.masked_factor = agent_data[1][12]
        agent_re.vaccinated_factor = agent_data[1][13]
        agent_re.time_in_class = agent_data[1][14]

        return agent_re

    elif uid[1] == Location.TYPE:
        if uid in agent_cache:
            agent_re = agent_cache[uid]
        else:
            agent_re = Location(uid[0], uid[2], agent_data[2][0])
            agent_cache[uid] = agent_re

        agent_re.all_info = agent_data[2][0]
        agent_re.real_coords = agent_data[2][1]
        agent_re.adjusted_coords = agent_data[2][2]
        agent_re.loc_type = agent_data[2][3]
        agent_re.agent_count = agent_data[2][4]
        agent_re.susceptible_count = agent_data[2][5]
        agent_re.exposed_count = agent_data[2][6]
        agent_re.infectious_count = agent_data[2][7]
        agent_re.recovered_count = agent_data[2][8]

        return agent_re

    

class Model:

    def __init__(self, comm, params, location_info, agents_info, params_sa):

        #Retrieves CPU rank from space to handle per the mpirun definition
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()

        #Creates the runner along with its step pattern and stopping point
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        #Defines the bounding box for the sim
        box = space.BoundingBox(params['min.longitude'], params['longitude.extent'], params['min.latitude'], params['latitude.extent'], 0, 0)

        #Defines the space that exists within the bounding box
        self.space = space.SharedCSpace('space', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                        buffer_size=2, comm=comm, tree_threshold=100)
        
        #Actually adds the space to the context
        self.context.add_projection(self.space)

        world_size = comm.Get_size()

        self.time_of_day = 1
        self.day_of_week = 1

        self.all_location_info = location_info
        self.all_agent_info = agents_info

        self.num_agents = len(agents_info)
        self.num_locations = len(location_info)

        agent_per_rank = int(self.num_agents / world_size)

        self.params_sa = params_sa
        
        self.isolation_rate = params_sa[0]
        self.basic_rep_ratio = params_sa[1]
        self.vaccine_factor = params_sa[2]
        self.mask_chance = params_sa[3]
        self.mask_factor = params_sa[4]

        #Create agents per the above values
        for i in range(agent_per_rank):

            h = Person(i + (self.rank * agent_per_rank), self.rank, agents_info[i + (self.rank * agent_per_rank)], self.params_sa)

            self.context.add(h)

            #Puts agents in their homes scaled accordingly
            x = agents_info[i + (self.rank * agent_per_rank)][1][0] * (-100000)
            y = agents_info[i + (self.rank * agent_per_rank)][1][1] * 100000

            self.space.move(h, cpt(x, y))
            
        location_per_rank = int(self.num_locations / world_size)

        #Creates an equal number of locations on each rank
        for i in range(location_per_rank):
            l = Location(i + (self.rank * location_per_rank), self.rank, location_info[i + (self.rank * location_per_rank)])
            self.context.add(l)

            x = location_info[i + (self.rank * location_per_rank)][3][0] * (-100000)
            y = location_info[i + (self.rank * location_per_rank)][3][1] * 100000

            self.space.move(l, cpt(x,y))

        #Restores agents on their proper rank after location-rank initialization
        self.context.synchronize(restore_agent)
            
        #Creates the loggers that will track all pieces of information necessary
        self.meet_log = MeetLog()
        loggers = logging.create_loggers(self.meet_log, op=MPI.SUM, names={'susceptible_count': 'susceptible'}, rank=self.rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.SUM, names={'exposed_count': 'exposed'}, rank=self.rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.SUM, names={'infectious_count': 'infectious'}, rank=self.rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.SUM, names={'recovered_count': 'recovered'}, rank=self.rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.SUM, names={'total_count': 'total'}, rank=self.rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.SUM, names={'masked_count': 'masked'}, rank=self.rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.SUM, names={'vaccinated_count': 'vaccinated'}, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, comm, params['meet_log_file'])

        for person_ag in self.context.agents(Person.TYPE):
            person_ag.count_states(self.meet_log)
        self.data_set.log(0)
        self.meet_log.susceptible_count = self.meet_log.exposed_count = self.meet_log.infectious_count = self.meet_log.recovered_count = self.meet_log.total_count = self.meet_log.masked_count = self.meet_log.vaccinated_count = 0

    #Runs the model
    def run(self):
        self.runner.execute()

    #This is what happens on each time step
    def step(self):

        self.move_agents()

        self.context.synchronize(restore_agent)

        self.update_loc()

        self.step_loc()

        self.update_time()

        self.log_all()

    #Move all agents to their next location
    def move_agents(self):

        for person_agent in self.context.agents(Person.TYPE):
            person_agent.move()

    #Update the info each location stores
    def update_loc(self):

        for location_agent in self.context.agents(Location.TYPE):

            if ((self.day_of_week % 2 == 1 and location_agent.loc_type == "home") or (self.day_of_week in (2,4,6,8,10) and location_agent.loc_type == "job")
            or (self.day_of_week in (12,14) and location_agent.loc_type == "store")):
                
                location_agent.update_info()

    #Step through all locations and spread disease
    def step_loc(self):

        for location_agent in self.context.agents(Location.TYPE):

            if ((self.day_of_week % 2 == 1 and location_agent.loc_type == "home") or (self.day_of_week in (2,4,6,8,10) and location_agent.loc_type == "job")
            or (self.day_of_week in (12,14) and location_agent.loc_type == "store")):
                
                location_agent.step()

    #Update the time of day from morning to night or night to morning, and the day of week
    def update_time(self):

        if self.time_of_day == 1:
            self.time_of_day = 2
        else:
            self.time_of_day = 1

        if self.day_of_week <= 13:
            self.day_of_week += 1
        else:
            self.day_of_week = 1

    #Write all current data to the logging file
    def log_all(self):

        for person_ag in self.context.agents(Person.TYPE):
            person_ag.count_states(self.meet_log)

        tick = self.runner.schedule.tick
        self.data_set.log(tick)
        # clear the meet log counts for the next tick
        self.meet_log.susceptible_count = self.meet_log.exposed_count = self.meet_log.infectious_count = self.meet_log.recovered_count = self.meet_log.total_count = self.meet_log.masked_count = self.meet_log.vaccinated_count = 0

    def at_end(self):
        self.data_set.close()


def run_sim(sim_mode, num_time_steps, params_sa, sa_mode, trial_info):

    info = Kingston_Info_Comb.main_kingston_geo(sim_mode)
    location_details = info[0]
    agent_details = info[1]

    min_x = 100000000
    min_y = 100000000
    max_x = 0
    max_y = 0

    for loc in location_details:

        if loc[3][0] * (-100000)  <= min_x:
            min_x = loc[3][0] * (-100000)
        if loc[3][1] * 100000 <= min_y:
            min_y = loc[3][1] * 100000
        if loc[3][0] * (-100000) >= max_x:
            max_x = loc[3][0] * (-100000)
        if loc[3][1] * 100000 >= max_y:
            max_y = loc[3][1] * 100000

    params = {

        "stop.at": (num_time_steps - 1),
        "min.longitude": int(np.floor(min_x)),
        "longitude.extent": int(np.ceil(max_x)) - int(np.floor(min_x)),
        "min.latitude": int(np.floor(min_y)),
        "latitude.extent": int(np.ceil(max_y)) - int(np.floor(min_y)),
        'meet_log_file': "output/" + str(sa_mode) + "_" + str(trial_info[0]) + "_" + str(trial_info[1]) + "_counts.csv"

    }

    global model
 
    model = Model(MPI.COMM_WORLD, params, location_details, agent_details, params_sa)
    model.run()

    