import random
from dataclasses import dataclass
from typing import Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters

##added by Linus
#from dg_commons.maps.lanes import DgLanelet, LaneCtrPoint
from typing import Tuple, List
import numpy as np
import shapely as sh
import shapely.geometry as geo

# #comment out bevore submitting
# from shapely.ops import unary_union, nearest_points
# import time
# import matplotlib.pyplot as plt
# import geopandas as gpd
# from datetime import datetime
# #import copy

#Node class to safe RRT tree of nodes
class Node:
    def __init__(self, node:Tuple[float, float], parent = None, cost_to_parent:float = None, cost_to_start:float = None) -> None:
        self._node:Tuple[float, float] = node #_node since private var
        self.parent = parent
        self.cost_to_parent:float = cost_to_parent
        self.cost_to_start:float = cost_to_start

    def node(self) -> Tuple[float, float]:
        return self._node

@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2

class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""
    def __init__(self,
                 sg: VehicleGeometry,
                 sp: VehicleParameters
                 ):
        self.sg = sg
        self.sp = sp
        self.name: PlayerName = None
        self.goal: PlanningGoal = None
        self.lanelet_network: LaneletNetwork = None
        self.static_obstacles: Sequence[StaticObstacle] = None
        
        ##added by Linus
        self.rrt_tree:List[Node] = None
        self.optimal_path_LineString:geo.LineString = None
    
    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator at the beginning of each episode."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.static_obstacles = list(init_obs.dg_scenario.static_obstacles.values())
        
        ## added by linus
        #only called once in the beginning, sees all static objects and informations
        self.dt:float = init_obs.dg_scenario.scenario.dt #simulation timestep
        self.obstacle_strtree = init_obs.dg_scenario.strtree_obstacles #strtree of the static obstacles

        # #comment out bevore submittingS
        # #unify the lanelet_network to one shapely polygon
        # self.clear_union = unary_union([lanelet.shapely_object for lanelet in self.lanelet_network.lanelet_polygons]) #shapely polygon with all lanelets unified as one polygon(later obstacles will be substracted)
        # #substract obstacles from clear_union and safe it in union_with_obstacles
        # self.union_with_obstacles = self.clear_union
        # for obs in self.static_obstacles:
        #     self.union_with_obstacles = self.union_with_obstacles.difference(obs.shape)
        # #add goal polygon to valid sample region
        # self.union_with_obstacles_goal = self.union_with_obstacles.union(self.goal.goal)
        # #rectangle around valid sample region
        # self.sample_bounds:Tuple = self.union_with_obstacles_goal.bounds # (minx, miny, maxx, maxy)

        #initialize controller
        self.ddelta_controller:PDcontroller = PDcontroller(15.0, 120.0, None) #tune PD parameters 10.0, 200.0

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """ This method is called by the simulator at each time step.
        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state

        :param sim_obs:
        :return:
        """
        #plan the initial path
        if sim_obs.time == 0.0:
            optimal_path:List[Node] = self.rrt((sim_obs.players[self.name].state.x, sim_obs.players[self.name].state.y))
            list_of_tuples:List = [node.node() for node in optimal_path]
            self.optimal_path_LineString:geo.LineString = geo.LineString(list_of_tuples)
            #feed path to controller
            self.ddelta_controller.path = self.optimal_path_LineString

            # #comment out bevore submitting
            # #plot optimal path points
            # p = gpd.GeoSeries(self.union_with_obstacles_goal)
            # p.plot()
            # x = [x_temp.node()[0] for x_temp in optimal_path]
            # y = [y_temp.node()[1] for y_temp in optimal_path]
            # plt.plot(x, y, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green", linestyle="--", color='red')
            # plt.show()
            # timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
            # plt.savefig("optimal_path"+timestr+".png")
            # time.sleep(2) #to generate plots

        #velocity
        desired_speed:float = 4.0
        agent_in_front:bool = False
        if self.other_agent_in_front(sim_obs):
            desired_speed = 1.0 #0.8 #1.0 to not get stuck in a dead end situation
            agent_in_front = True
        cmd_acc:float = self.ddelta_controller.control_acc(des_speed=desired_speed, cur_velocity=sim_obs.players[self.name].state.vx, acc_kp=3.5, acc_kd=1.5, agent=self) #acc_kp=5.0 acc_kd=1.0 #must acc_kd < acc_kp
        #steering
        shift_direction:float = sim_obs.players[self.name].state.psi
        shift_x:float = self.sg.length * np.cos(shift_direction) #shift reference point from center to nose of the car if /2.0 otherwise even infront of the nose
        shift_y:float = self.sg.length * np.sin(shift_direction)
        shifted_position:geo.Point = geo.Point((sim_obs.players[self.name].state.x+shift_x, sim_obs.players[self.name].state.y+shift_y))
        cmd_ddelta:float = self.ddelta_controller.control_ddelta(shifted_position, sim_obs.players[self.name].state.delta, self.collision_steering(sim_obs, agent_in_front), self)
        
        return VehicleCommands(acc=cmd_acc, ddelta=cmd_ddelta)

    ##added by Linus

    #PRE:
    #POST: return true if an other agent in front of the vehicle is detected
    def other_agent_in_front(self, sim_obs:SimObservations):
        shifted_coords:List = list(sim_obs.players[self.name].occupancy.exterior.coords)
        #shift own occupancy to front to only stop when an object is in front
        shift_distance:float = self.sg.length
        shift_direction:float = sim_obs.players[self.name].state.psi - 0.2#minus to shift almost 30° (0.5rad) to the right #TODO tune angle of shifting
        shift_x:float = shift_distance * np.cos(shift_direction) #shift to front to don't care if somebody hits in the back and drive away (escape collision)
        shift_y:float = shift_distance * np.sin(shift_direction)
        #shift coords
        for idx, coord in enumerate(shifted_coords):
            shifted_coords[idx] = (coord[0]+shift_x, coord[1]+shift_y)
        lidar_self = geo.Polygon(shifted_coords)
        lidar_self = lidar_self.buffer(self.sg.w_half)
        for idx, coord in enumerate(shifted_coords):
            shifted_coords[idx] = (coord[0]+shift_x, coord[1]+shift_y) #shift the polygon again to the front
        lidar_self = lidar_self.union(geo.Polygon(shifted_coords)) #look further to the front
        #check if possible collision for all agents in sim_obs
        for playername in list(sim_obs.players.keys()):
            if playername == self.name:
                continue
            lidar_other = sim_obs.players[playername].occupancy
            if lidar_self.intersects(lidar_other): # and self.name > playername:
                return True
    
        return False

    #PRE:
    #POST: steers to the left as long as a possible collision (on the right) is detected and no obstacle is at the left
    def collision_steering(self, sim_obs:SimObservations, possible_colllision:bool) -> bool:
        if not possible_colllision:
            return False #dont steer to the left
        #shift own occupancy to front left
        shifted_coords:List = list(sim_obs.players[self.name].occupancy.exterior.coords)
        shift_direction:float = sim_obs.players[self.name].state.psi + 0.25#plus to shift almost 30° (0.5rad) to the left
        shift_x:float = self.sg.length * np.cos(shift_direction)
        shift_y:float = self.sg.length * np.sin(shift_direction)
        #shift coords
        for idx, coord in enumerate(shifted_coords):
            shifted_coords[idx] = (coord[0]+shift_x, coord[1]+shift_y)
        shifted_own_occupancy = geo.Polygon(shifted_coords)
        #only steer to the left if no other car is on the left
        for playername in list(sim_obs.players.keys()):
            if playername == self.name:
                continue
            if shifted_own_occupancy.intersects(sim_obs.players[playername].occupancy):
                return False
        #check possible collision with obstacle
        if any(obstacle.shape.intersects(shifted_own_occupancy) for obstacle in self.static_obstacles): #occupied on the left
            return False #dont steer to the left

        # #comment out bevore submitting
        # print(self.name, "collision_steering")

        return True
        
    #PRE:
    #POST: returns the cost of a single node to node connection
    def cost_dist(self, cur_node:Node) -> float:
        return np.sqrt((cur_node.node()[0]-cur_node.parent.node()[0])**2 + (cur_node.node()[1]-cur_node.parent.node()[0])**2) #eucledian distance

    #PRE: shapely Point
    #POST: returns the Node of self.rrt_tree that is closest to the Point
    def nearest_node(self, point:geo.Point) -> Node:
        closest_node:Node = Node((float("inf"), float("inf")), None) #initialize with node that is clearly out of grid
        for node in self.rrt_tree:
            if point.distance(geo.Point(node.node()[0], node.node()[1])) < point.distance(geo.Point(closest_node.node()[0], closest_node.node()[1])):
                closest_node = node
        return closest_node

    #PRE:
    #POST: make a step towards the random sampled point
    def step_to_random_point(self, nearest:Node, random:geo.Point, eta:float) -> Node:
        dist:np.ndarray = np.array((random.x-nearest.node()[0], random.y-nearest.node()[1]))
        dist = dist * eta
        return Node((nearest.node()[0]+dist[0], nearest.node()[1]+dist[1]), nearest)

    #PRE: current Node to check
    #POST: retruns True if the straight path to it's parent is collision free / is fully in the boundaries of the LaneletNetwork
    #TODO make efiicient since called many times during rrt
    def collisionfree_path(self, cur_point:Node) -> bool:
        #Connect current node to parent
        connection:geo.LineString = geo.LineString([cur_point.node(), cur_point.parent.node()])
        
        obstacles_and_borders = self.obstacle_strtree.query(connection.buffer(distance=self.sg.width+0.24, cap_style=3)) #buffer with carwidth to avoid collisions
        for element in obstacles_and_borders:
            if element.intersects(connection.buffer(distance=self.sg.width+0.24, cap_style=3)): #.buffer(distance=self.sg.width, cap_style=3)
                return False
        #return True
        return not self.sharp_turn(cur_point)

    #PRE:
    #POST: returns true if the new node would lead to a sharp turn (less than 90°)
    def sharp_turn(self, cur_node:Node) -> bool:
        if cur_node.parent.parent == None: #first connection cannot have an angle
            return False
        cur_x: float = cur_node.node()[0] - cur_node.parent.node()[0]
        cur_y: float = cur_node.node()[1] - cur_node.parent.node()[1]
        parent_x:float = cur_node.parent.parent.node()[0] - cur_node.parent.node()[0]
        parent_y:float = cur_node.parent.parent.node()[1] - cur_node.parent.node()[1]
        # if cur_x*parent_x + cur_y*parent_y > 0.0: #use dot product (dot product of 90° == 0.0)
        #     return True
        cur:np.ndarray = np.array([cur_x, cur_y])
        parent:np.ndarray = np.array([parent_x, parent_y])
        angle:float = np.arccos(np.dot(cur, parent)/np.linalg.norm(cur)/np.linalg.norm(parent)*0.9999)
        if angle == np.NAN or angle < 1.65: #90° == 1.5708rad
            return True
        return False

    #PRE:
    #POST: returns true if the node is in the goal
    def node_in_goal(self, node:Node):
        if self.goal.goal.buffer(self.sg.w_half).contains(geo.Point(node.node()[0], node.node()[1])): #buffer goal to reach it more easily since probably I'll hit it anyway
            return True
        return False

    #PRE:
    #POST: reconnected the close(7.0) nodes in self.rrt_tree if they can be reached with less cost trough the new_node
    def star(self, new_node:Node) -> List[Node]:
        for node in self.rrt_tree:
            temp_node:Node = Node((node.node()[0], node.node()[1]), new_node)
            cost_temp:float = self.cost_dist(temp_node)
            if cost_temp<5.0 and node.cost_to_start > new_node.cost_to_start+cost_temp and self.collisionfree_path(temp_node): #7.0
                node.parent = new_node
            
        return self.rrt_tree
        
    #PRE:
    #POST: returns cost of the node back to start
    def cost_to_start(self, node:Node) -> float:
        node_iterator:Node = node
        temp_cost:float = 0.0
        while node_iterator is not None:
            temp_cost += node_iterator.cost_to_parent
            node_iterator = node_iterator.parent
        return temp_cost

    #PRE:
    #POST: retruns the node of self.rrt_tree that provides the shortest cost to start when attaching the node
    def find_lowest_cost_to_start_parent(self, node:Node) -> Node:
        low_cost_parent:Node = node.parent
        temp_cost:float = node.cost_to_start
        for tree_node in self.rrt_tree:
            temp_node:Node = Node(node.node(), tree_node)
            temp_node.cost_to_parent = self.cost_dist(temp_node)
            temp_node.cost_to_start = self.cost_to_start(temp_node)
            if temp_node.cost_to_start < temp_cost and self.collisionfree_path(temp_node):
                low_cost_parent = tree_node
                temp_cost = temp_node.cost_to_start
        return low_cost_parent

    #PRE: goal_node with valid parents
    #POST: returns the optimal path from start to goal
    def optimal_path(self, goal_node:Node) -> List[Node]:
        path:List = []
        node_iterator:Node = goal_node
        while node_iterator is not None:
            path.append(node_iterator)
            node_iterator = node_iterator.parent
        path.reverse() #make list from start to goal
        return path

    #PRE:
    #POST: returns a tuple of a random sampled point in the hardcoded AABB with a bias to sample more in the goal and hard acces areas
    def sample_biased_points(self, goal:geo.Point) -> Tuple[float,float]:
        prob_to_sample_goal:float = 0.15
        prob_to_sample_hardaccess:float = 0.05
        generator:float = random.uniform(0.0, 1.0)
        if generator > 1.0-prob_to_sample_goal:
            return (goal.x+random.uniform(-2.0, 2.0), goal.y+random.uniform(-2.0, 2.0)) #sample goal (jitter around it)
        hardaccess1:Tuple[float,float] = (-27.0, -20.0) #goal in lower left corner
        hardaccess2:Tuple[float,float] = (-20.0, 0.0) #goal in lower left corner
        hardaccess3:Tuple[float,float] = (-20.0, -10.0) #goal in lower left corner
        hardaccess4:Tuple[float,float] = (-28.0, -25.0) #goal in lower left corner
        hardaccess5:Tuple[float,float] = (-12.0, 12.5) #goal in lower left corner
        hardaccess6:Tuple[float,float] = (19.0, -1.0) #goal on the right correct heading
        hardaccess7:Tuple[float,float] = (23.0, 10.0) #goal on the right wrong heading
        hardaccess71:Tuple[float,float] = (11.0, 13.0) #goal on the right wrong heading
        hardaccess8:Tuple[float,float] = (-19.0, 28.0) #goal on the left correct heading
        goal_hard_to_access:bool = False
        hard_sample:Tuple[float,float] = []
        if goal.x < -25.0 and goal.y < -20.0 and generator < prob_to_sample_hardaccess: #goal in lower left corner
            goal_hard_to_access = True
            hard_sample = random.choice([hardaccess5, hardaccess2, hardaccess3, hardaccess1, hardaccess4])
        elif goal.x > 20.0 and goal.y < -5.0 and generator < prob_to_sample_hardaccess:#goal on the right correct heading
            goal_hard_to_access = True
            hard_sample = hardaccess6
        elif goal.x > 36.0 and goal.y > -4.0 and generator < prob_to_sample_hardaccess:#goal on the right wrong heading
            goal_hard_to_access = True
            hard_sample = random.choice([hardaccess7, hardaccess71])
        elif goal.x < -26.0 and goal.y > 37.0 and generator < prob_to_sample_hardaccess:#goal on the left correct heading
            goal_hard_to_access = True
            hard_sample = hardaccess8

        return hard_sample if goal_hard_to_access else (random.uniform(-44, 40.0), random.uniform(-46.0, 75.0))

    #PRE:
    #POST: returns nodes containing a path that leads from start to goal
    def rrt(self, start_pos:Tuple[float, float]) -> List[Node]:
        #init vars
        goal_point:geo.Point = self.goal.goal.centroid #create a point in the middle of the goal
        self.rrt_tree = [Node(node=start_pos, parent=None, cost_to_parent=0.0, cost_to_start=0.0)] #add starting node to tree
        iterations = 1500 #number of sampling iterations

        # #comment out bevore submitting
        # p = gpd.GeoSeries(self.union_with_obstacles_goal)
        # p.plot()

        #sample for n iterations or until connection to goal can be made
        for it in range(iterations):
            
            #comment in bevore submitting
            #random.seed(it*6)
            random.seed(it*3) #random.seed(it**2) #random.seed(it*3) #random.seed(it*2) #random.seed(it+2) #random.seed(it+1)

            # if it == iterations-1:
            #     print("PROBABLY LAST ITERATION REACHED")
            random_point_in_bound:Tuple[float, float] = self.sample_biased_points(goal_point)
            random_point_shapely:geo.Point = geo.Point(random_point_in_bound[0], random_point_in_bound[1]) #Convert to shapely object
            #find nearest node
            near_parent:Node = self.nearest_node(random_point_shapely)
            #create current node
            current_new:Node = Node(node=random_point_in_bound, parent=near_parent)
            #don't take nodes that are too far
            current_new.cost_to_parent = self.cost_dist(current_new)
            current_new.cost_to_start = self.cost_to_start(current_new)
            if current_new.cost_to_parent > 15.0: #20.0
                current_new = self.step_to_random_point(near_parent, random_point_shapely, 0.2)
                current_new.cost_to_parent = self.cost_dist(current_new)
                current_new.cost_to_start = self.cost_to_start(current_new)
            #point should not be outside of the road
            in_lanelet:bool = False
            if self.goal.goal.contains(geo.Point(current_new.node()[0], current_new.node()[1])):
                in_lanelet = True
            elif any(lanelet._shapely_polygon.contains(geo.Point(current_new.node()[0], current_new.node()[1])) for lanelet in self.lanelet_network.lanelet_polygons):
                in_lanelet = True
            if not in_lanelet:
                continue
            #see if a shorter route than the nearest node can be taken
            current_new.parent = self.find_lowest_cost_to_start_parent(current_new)
            current_new.cost_to_parent = self.cost_dist(current_new)
            current_new.cost_to_start = self.cost_to_start(current_new)
            #add newly sampled node to the tree if no collision is encountered
            if self.collisionfree_path(current_new):
                self.star(current_new)
                self.rrt_tree.append(current_new)
                if self.node_in_goal(current_new):
                    break

                # #comment out bevore submitting
                # plt.plot([current_new.node()[0], current_new.parent.node()[0]], [current_new.node()[1], current_new.parent.node()[1]], marker="o", markersize=3, markeredgecolor="red", linestyle="--", color='red')
                
                #Check if a collision free connection can be made to the goal if so exit, since a path is found
                #TODO maybe ditch this Idea intirely since I'm now also sampling the Goal
                #create goal node
                goal_node:Node = Node(node=(goal_point.x, goal_point.y), parent=current_new)
                goal_node.cost_to_parent = self.cost_dist(goal_node)
                if goal_node.cost_to_parent > 30.0: #60.0
                    continue
                if self.collisionfree_path(goal_node):
                    self.rrt_tree.append(goal_node)
                    break
        
        # #comment out bevore submitting
        # #plot wholetree
        # plt.show()
        # timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
        # plt.savefig("wholetree"+timestr+".png")

        goal_node:Node = self.rrt_tree[-1]

        return self.optimal_path(goal_node)

class PDcontroller:
    def __init__(self, kp:float, kd:float, path:geo.LineString) -> None:
        self.kp:float = kp
        self.kd:float = kd
        self.path:geo.LineString = path
        self.last_error:float = 0.0
        self.last_error_acc:float = 0.0
        self.psi_of_closest_path:float = 0.0
        self.emergency_intervention_done:int = 0
    #PRE:
    #POST: Returns True/False depending on which side of the path the bicycle is
    def left_of_path(self, cur_pos:geo.Point) -> bool:
        min_dist:float = float("inf")
        closest_segment:geo.LineString = None
        for node_idx in range(len(self.path.coords)-1): #TODO change to enumerate
            segment:geo.LineString = geo.LineString([self.path.coords[node_idx], self.path.coords[node_idx+1]]) #change to enumerate element
            dist:float = segment.distance(cur_pos)
            if dist < min_dist:
                min_dist = dist
                closest_segment = segment
        x_line:float = closest_segment.coords[1][0] - closest_segment.coords[0][0]
        y_line:float = closest_segment.coords[1][1] - closest_segment.coords[0][1]
        self.psi_of_closest_path = np.arctan2(y_line, x_line)
        line:np.ndarray = [x_line, y_line, 0.0] #vector in the direction of the closest path segment
        car:np.ndarray = [cur_pos.x - closest_segment.coords[0][0], cur_pos.y - closest_segment.coords[0][1], 0.0] #vector from the start of the closest segment to the car
        side:np.ndarray = np.cross(line, car)

        if side[2] < 0.0:
            return True
        return False

    #PRE:
    #POST: ddelta to follow self.path
    def control_ddelta(self, cur_pos:geo.Point, cur_delta:float, collision_avoid_psi:float, agent:Pdm4arAgent) -> float:
        if collision_avoid_psi: #emergency steering necessary to avoid agent-agent collision (drive in a fixed direction on the map)
            if self.emergency_intervention_done < 2:
                self.emergency_intervention_done += 1
                return agent.sp.ddelta_max
            return 0.0

            # #comment out bevore submitting
            # print(agent.name, "delta", delta, "delta_error", delta_error)
            # print(sim_obs.time, "Lanelet", self.psi_of_closest_path, "sim_obs psi", sim_obs.players[agent.name].state.psi)
        
        #reset emergency intervention
        self.emergency_intervention_done = 0

        #closest point on path to current vehicle position
        closest_point_LineString:geo.Point = self.path.interpolate(self.path.project(cur_pos)) #maybe don't take center position of car but nose position
        #error (bigger the more away from path)
        error = closest_point_LineString.distance(cur_pos)
        #correct the sign of the error depending on which side of the path the bicycle is
        if self.left_of_path(cur_pos):
            error = -error
        error_derivatie = (error - self.last_error) / agent.dt if not agent.dt == 0.0 else 0.0
        #PD formula
        delta:float = self.kp*error + self.kd*error_derivatie
        #cascade
        delta_error = delta-cur_delta

        #for next iteration
        self.last_error = error
        #clip ddelta
        if delta_error > 0.0:
            ddelta = np.min([delta_error, agent.sp.ddelta_max])
        else:
            ddelta = np.max([delta_error, -agent.sp.ddelta_max])

        #no steering if already at full steering
        if np.abs(cur_delta) >= agent.sp.delta_max:
            if ddelta < 0.0 and cur_delta < 0.0:
                ddelta = 0.0
            elif ddelta > 0.0 and cur_delta > 0.0:
                ddelta = 0.0
        
        return -ddelta

    #PRE: desired speed
    #POST: acc to follow desired speed
    def control_acc(self, des_speed:float, cur_velocity:float, acc_kp:float, acc_kd:float, agent:Pdm4arAgent) -> float:
        error_acc:float = des_speed - cur_velocity
        error_derivatie:float = 0.0
        if error_acc > 0.0: #break instantly, but accelerate slowly
            error_derivatie = (error_acc - self.last_error_acc) / agent.dt if not agent.dt == 0.0 else 0.0
        #PD and cascade
        acc = error_acc*acc_kp + error_derivatie*acc_kd
        #for next iteration
        self.last_error_acc = error_acc
        #clip acc
        if acc > agent.sp.acc_limits[1]:
            acc = agent.sp.acc_limits[1]
        elif acc < agent.sp.acc_limits[0]:
            acc = agent.sp.acc_limits[0]
        return acc
