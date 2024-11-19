# Disease_Models
Geographical agent-based disease models based in Mesa, Repast, RDDL, and NetLogo. The model works as follows for all four models:

Each agent has a home, job, and two store locations. Each day is broken up into two time steps - one where agents are at their homes, and one where agents are at their job or a store (depending on their age, if they are a student, and if they are self-isolating). On every time step, agents who are infectious have a probability of spreading disease to other agents at that location based on the basic reproduction number (user-defined) and the density of agents at the location. When an agent contracts the disease, they have some user-defined probability of self-isolating (user-defined), where they remain at their home on all time steps and only spread disease to agents at their home. Agents can also mask and vaccinate. The probability of masking is user-defined and the probability of vaccinating is based on the agent's age, derived from data related to COVID-19. The efficacy of masking and vaccinating is user-defined. 

These agent-based models simulate disease based on the SEIR compartmental model, where agents are in one of four compartments (susceptible, exposed, infectious, and recovered), beginning as susceptible by default, and flowing through them linearly when infectious (apart from the initially infectious agents defined at time step 0, who begin in the infectious compartment). Agents can spend any amount of time in the suscpetible compartment until infected, at which point they exist in the other 3 compartments one by one for some normally distributed amount of time specific to each compartment.

### RDDL Notes
The RDDL directory is subdiveded into two directories - General and RDDPS. The General model is a RDDL model that does not make use of the planner in any way - it is simply an agent-based model that can be readily adapted to perform some planning task. The RDDPS model makes direct use of the planner by having it implement mask/vaccine mandates on the population with the aim of minimizing mandate uptime while minimizing the number of hospitalizations over some user-defined capacity. The problem files given alongside both domains were generated based on Kingston Geography, with about 100 agents.


Disease_Models © 2024 by Bruce Chidley is licensed under Creative Commons Attribution 4.0 International 
