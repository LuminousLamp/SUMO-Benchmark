from pathlib import Path
import sys
sys.path.append(".")
from src.utils.read_data import read_data
from src import SUMOBaseline
from src.utils import SUMOBaselineVisualizer

# FIXME: replace the path to a WOMD tfrecord shard file
scenario_list = read_data(
    "/media/led/WD_2TB/WOMD/validation/validation.tfrecord-00000-of-00150",
)

for i, scenario in enumerate(scenario_list):

    scenario_id = scenario.scenario_id
    # FIXME: fill in the path to the generated SUMO .net.xml file and an appropriately configured .sumocfg file
    base_dir = Path(f"/media/led/WD_2TB/Dataset/maps/waymo-validation/")
    sumo_cfg_path = Path(base_dir / f"{scenario_id}/{scenario_id}.sumocfg")
    sumo_net_path = Path(base_dir / f"{scenario_id}/{scenario_id}.net.xml")

    # instantiate a SUMOBaseline class, call .run_simulation() to run the simulation once
    # adjust sim_wallstep to the desired number of simulation steps (10Hz)
    simulator = SUMOBaseline(scenario, sumo_cfg_path, sumo_net_path, sim_wallstep=91)
    agent_attrs, agent_states, one_sim_time_buff_agents, one_sim_time_buff_tls = simulator.run_simulation(
        offroad_mapping=True,
        gui=False,
        sumo_oracle_parameters={}
    )
    # you can do more configurations within sumo_oracle_parameters dictionary,
    # for more details on this, refer to files in ./configs
    
    # the simulation returns a (agent_attrs, agent_states) tuple
    # agent_attrs is a dictionary of {agent_id: AgentAttr()}
    # agent states is a dictionary of {agent_id: [AgentState()]}
    # for more information on AgentAttr and AgentState, see ./src/dynamic_behavior/agents.py

    # you can visualize the simulation result using SUMOBaselineVisualizer()
    visualizer = SUMOBaselineVisualizer(scenario, shaded=True, types_to_draw="withedge", simplify_agent_color=False)
    visualizer.save_map(base_dir=".",) # save the map picture
    visualizer.save_video(
        one_sim_time_buff_agents,
        one_sim_time_buff_tls,
        base_dir=".",
        video_name=f"{scenario_id}.mp4",
        with_ground_truth=False
    ) # save the simulation as a video