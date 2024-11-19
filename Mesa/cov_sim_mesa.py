import mesa

import numpy as np
import pandas as pd

import Kingston_Info_Comb

#Agent class
#Stores what class they are in, and their mask/vaccination status
#Also tracks how long they have been in each class and how many agents they have spread disease to
class CovidAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, agent_info_all, model):
        #A child of the model class
        super().__init__(unique_id, model)

        self.age_bracket = agent_info_all[0][2]

        self.store_1 = agent_info_all[0][5]
        self.store_2 = agent_info_all[0][6]

        self.isolating = False

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
        if (prob_masked <= self.model.mask_chance):
            self.masked_factor = self.model.mask_factor
        else:
            self.masked_factor = 1

        prob_vaccinated = np.random.uniform(0,1)

        #Set to 1 by default
        self.vaccinated_factor = 1

        #Vaccination rates based on age bracket
        if self.age_bracket == 0:
            if (prob_vaccinated <= 0.251):
                self.vaccinated_factor = self.model.vaccine_factor
        elif self.age_bracket == 1:
            if (prob_vaccinated <= 0.771):
                self.vaccinated_factor = self.model.vaccine_factor
        elif self.age_bracket == 2:
            if (prob_vaccinated <= 0.819):
                self.vaccinated_factor = self.model.vaccine_factor
        elif self.age_bracket == 3:
            if (prob_vaccinated <= 0.851):
                self.vaccinated_factor = self.model.vaccine_factor
        elif self.age_bracket == 4:
            if (prob_vaccinated <= 0.883):
                self.vaccinated_factor = self.model.vaccine_factor
        elif self.age_bracket == 5:
            if (prob_vaccinated <= 0.885):
                self.vaccinated_factor = self.model.vaccine_factor
        elif self.age_bracket == 6:
            if (prob_vaccinated <= 0.940):
                self.vaccinated_factor = self.model.vaccine_factor
        elif self.age_bracket == 7:
            if (prob_vaccinated <= 0.982):
                self.vaccinated_factor = self.model.vaccine_factor
        elif self.age_bracket == 8:
            if (prob_vaccinated <= 0.990):
                self.vaccinated_factor = self.model.vaccine_factor

        self.time_in_class = 0

        self.spread_to = 0

    #When the scheduler takes a step, agents do nothing. The agent behaviour is performed in the "location" agents
    def step(self):
        
        pass


#Location class is where the bulk of the mechanics lie
#Each location is stepped through, and on each step, every agent at a given location has their status updated based on other agents at that location
class CovidLocation(mesa.Agent):

    def __init__(self, unique_id, loc_info, agents_belonging, model):
        #Child of the model class
        super().__init__(unique_id, model)
        
        #Extracts some key info about the location
        self.loc_type = loc_info[1]
        self.loc_id = unique_id
        self.agents_at_loc = agents_belonging

        #Sets the initial values for number of agents of a given class at a location
        self.susceptible_count = 0
        self.exposed_count = 0
        self.infectious_count = 0
        self.recovered_count = 0
        for item in self.agents_at_loc:
            if item.susceptible:
                self.susceptible_count += 1
            elif item.exposed:
                self.exposed_count += 1
            elif item.infectious:
                self.infectious_count += 1
            else:
                self.recovered_count += 1

    #Updates the class of a single agent
    def update_class(self, current_agent):

        #This will be set to "True" if an agent is successfully infected by another agent
        newly_exposed = False

        #Captures the mechanics of one agent being exposed to the disease by another
        if current_agent.susceptible:
            
            #For all agents at the same location as the agent in question, test to see if they infect the agent in question
            for agent_at_same_loc in self.agents_at_loc:


                if (self.model.day_of_week == 12 and agent_at_same_loc.store_1 == self.loc_id) or (self.model.day_of_week == 14 and agent_at_same_loc.store_2 == self.loc_id) or (not (self.model.day_of_week in (12, 14))):

                    #If an agent is infectious and not equalt to the current agent (although since the agent cannot be both susceptible and infectious, this would never happen)
                    if ((agent_at_same_loc.infectious) and (agent_at_same_loc != current_agent) and (not agent_at_same_loc.isolating)):

                        #The odds of an agent contracting the disease from another agent is (R0/(time in infectious class)) / (# of susceptibles at the location)
                        #This makes it so that each agent will infect approximately 3.32 agents, lining up with the definition of R0
                        #We multiply by the mask and vaccination factors if applicable for the agents in question
                        #Note that the infectious agent does not have their infectivity reduced by being vaccinated - if they are infected, they are spreading it the same (this is an assumption)
                        odds = (agent_at_same_loc.masked_factor * current_agent.masked_factor) * (current_agent.vaccinated_factor) * (self.model.basic_rep_ratio / 16) / (self.susceptible_count)

                        #If the infection is successful, then the agent in question becomes infected
                        against = np.random.uniform(0,1)
                        if (against < odds):
                            #Add 1 to the number of agents infected
                            agent_at_same_loc.spread_to += 1
                            newly_exposed = True
            
            #Update the agent in question's status and reset their class timer
            if newly_exposed:
                #print("Success!")
                current_agent.susceptible = False
                current_agent.exposed = True
                current_agent.time_in_class = 0

                #Set the time the agent will be in the exposed class
                current_agent.total_time_in_class = np.random.normal(9, 2)

        #If the agent in question is instead exposed and at the end of their exposd period, update their status and reset their class timer
        elif current_agent.exposed and current_agent.time_in_class >= current_agent.total_time_in_class:
            current_agent.exposed = False
            current_agent.infectious = True
            current_agent.time_in_class = 0

            #Set the time the agent will be in the infectious class
            current_agent.total_time_in_class = np.random.normal(16, 4)

            #Check to see if agent isolates
            isolating_prob = np.random.uniform(0, 1)
            if isolating_prob <= self.model.isolation_rate:
                current_agent.isolating = True

        #Same for infectious
        #Update the model.spread_count_list list with the total number of agents they successfully infected
        elif current_agent.infectious and current_agent.time_in_class >= current_agent.total_time_in_class:
            current_agent.infectious = False
            current_agent.recovered = True
            #print("Agent " + str(current_agent.unique_id) + " spread to " + str(current_agent.spread_to) + " agents")
            current_agent.model.spread_count_list.append(current_agent.spread_to)
            current_agent.spread_to = 0
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


    #Stepping through the locations
    def step(self):

        #First, do a count of the number of agents in each class present at a location
        #This must be done first, because agents are technically at all locations at once, and so if an agent is at locations x and y, but location x updates the status of the agent, its status might not be reflected in the current count for location y
        agent_list_here = []

        susceptible_per = 0
        exposed_per = 0
        infectious_per = 0
        recovered_per = 0

        for item in self.agents_at_loc:

            if (item.susceptible):
                susceptible_per += 1
            elif (item.exposed):
                exposed_per += 1
            elif (item.infectious):
                infectious_per += 1
            else:
                recovered_per += 1

        self.susceptible_count = susceptible_per
        self.exposed_count = exposed_per
        self.infectious_count = infectious_per
        self.recovered_count = recovered_per

        #We track the number of agents in each class once more after updating their classes for the sake of accurate reporting on each step
        #If we were to NOT care about the progress of the disease at each time step, we could theoretically just update it once at the start of the step() function
        susceptible_per = 0
        exposed_per = 0
        infectious_per = 0
        recovered_per = 0

        #For each agent at the location, update their status if it is the location's turn to step (dependent on the time of day)
        for item in self.agents_at_loc:

            if ((self.model.day_of_week % 2 == 1 and self.loc_type == "home") or (self.model.day_of_week in (2,4,6,8,10) and self.loc_type == "job")
            or (self.model.day_of_week == 12 and self.loc_type == "store" and item.store_1 == self.loc_id) or (self.model.day_of_week == 14 and self.loc_type == "store" and item.store_2 == self.loc_id)):

                #Update the agent's class and the time that they have been in the class
                self.update_class(item)
                item.time_in_class += 1

            #Append the agent and their class info
            agent_list_here.append((item.unique_id, item.susceptible, item.exposed, item.infectious, item.recovered))

            if (item.susceptible):
                susceptible_per += 1
            elif (item.exposed):
                exposed_per += 1
            elif (item.infectious):
                infectious_per += 1
            else:
                recovered_per += 1
        
        self.susceptible_count = susceptible_per
        self.exposed_count = exposed_per
        self.infectious_count = infectious_per
        self.recovered_count = recovered_per


#Model class, responsible for creating and stepping through all agents
class CovidModel(mesa.Model):


    def __init__(self, agents_info, location_info, params):
        #Extracts some basic info
        self.num_agents = len(agents_info)
        self.num_locations = len(location_info)
        self.params = params
        
        self.isolation_rate = params[0]
        self.basic_rep_ratio = params[1]
        self.vaccine_factor = params[2]
        self.mask_chance = params[3]
        self.mask_factor = params[4]

        #Creates the schedule - agents are stepped through randomly, although it does not matter in our case
        self.schedule = mesa.time.RandomActivation(self)

        self.running = True
        
        #Initialize to 1, indicating a Monday where all agents are at home
        self.day_of_week = 1

        #Total class counts are set to 0
        self.total_sus_count = 0
        self.total_exp_count = 0
        self.total_inf_count = 0
        self.total_rec_count = 0

        self.spread_count_list = []

        #Warm up the agent class?
        a = CovidAgent(agents_info[0][0][0], agents_info[0], self)

        agent_list = []

        self.all_agents = []

        masked_count = 0
        vaccinated_count = 0

        # Create agents
        for i in range(0, self.num_agents):
            a = CovidAgent(agents_info[i][0][0], agents_info[i], self)
            self.all_agents.append(a)
            agent_list.append((agents_info[i][0][0], a))
            self.schedule.add(a)

            if (a.masked_factor < 1):
                masked_count += 1

            if (a.vaccinated_factor < 1):
                vaccinated_count += 1

        self.total_masked_count = masked_count
        self.total_vaccinated_count = vaccinated_count
        
        #Create location agents with agent agents as objects of each location agent if the agent agent is at the location agent
        for i in range (0, self.num_locations):
            agents_at_loc = []

            for agent_temp in agent_list:
                if agent_temp[0] in location_info[i][2]:
                    agents_at_loc.append(agent_temp[1])

            l = CovidLocation(location_info[i][0], location_info[i], agents_at_loc, self)
            self.schedule.add(l)


    #Step through the model (not stepped through by the scheduler)
    def step(self):
        #self.datacollector.collect(self)

        #Steps through location and agent agents
        self.schedule.step()

        #If we have reached Sunday night, reset to Monday morning
        if (self.day_of_week <= 13):
            self.day_of_week += 1
        else:
                self.day_of_week = 1

        #Count the number of agents in each class on each time step
        sus_count = 0
        exp_count = 0
        inf_count = 0
        rec_count = 0

        for ag in self.all_agents:

            if (ag.susceptible):
                sus_count += 1
            elif (ag.exposed):
                exp_count += 1
            elif (ag.infectious):
                inf_count += 1
            elif (ag.recovered):
                rec_count += 1

        self.total_sus_count = sus_count
        self.total_exp_count = exp_count
        self.total_inf_count = inf_count
        self.total_rec_count = rec_count


def run_sim(sim_mode, num_time_steps, params):

    susceptible_counts = []
    exposed_counts = []
    infectious_counts = []
    recovered_counts = []
    time_steps = []

    info = Kingston_Info_Comb.main_kingston_geo(sim_mode, 0, 0)
    location_details = info[0]
    agent_details = info[1]

    #params[0]: Isolation Rate
    #params[1]: Basic Reproductive Ratio
    #params[2]: Vaccine Factor
    #params[3]: Mask Chance
    #params[4]: Mask Factor

    model = CovidModel(agent_details, location_details, params)
    model.reset_randomizer()

    for i in range(num_time_steps):

        model.step()

        susceptible_counts.append(model.total_sus_count)
        exposed_counts.append(model.total_exp_count)
        infectious_counts.append(model.total_inf_count)
        recovered_counts.append(model.total_rec_count)
        time_steps.append(i)

    counts_dict = pd.DataFrame({
        "susceptible_count": susceptible_counts,
        "exposed_count": exposed_counts,
        "infectious_count": infectious_counts,
        "recovered_count": recovered_counts,
        "time_step": time_steps,
        "masked_count": model.total_masked_count,
        "vaccinated_count": model.total_vaccinated_count
    })

    return counts_dict