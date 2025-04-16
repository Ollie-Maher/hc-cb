# Script to generate a environments for reinforcement learning

import minigrid
import minigrid.wrappers
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj, Goal, Wall, Floor, Lava
from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)


class T_Maze(MiniGridEnv):
    def __init__(
        self,
        size=7,
        agent_start_pos=(1, 3),
        agent_start_dir=0,
        max_steps: int | None = 100,
        task_switch: int = 1,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.floor_colours = ['red','blue']

        self.goal_up = True

        mission_space = MissionSpace(mission_func=self._gen_mission)

        self.task_switch = task_switch
        self.task_count = 0

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Choose left or right based on starting information"
    

    def _gen_grid(self, width, height):
        # Gen grid
        self.grid = Grid(width, height)
        # Walls
        self.grid.wall_rect(0, 1, width, height-2)
        self.grid.set(4,2,Wall())
        self.grid.set(4,4,Wall())
        '''
        WALLS:
        .......
        XXXXXXX
        X...X.X
        X.....X
        X...X.X
        XXXXXXX
        .......
        '''
        # Coloured floors
        self.grid.set(1,2,Floor(self.floor_colours[self.goal_up]))
        self.grid.set(1,4,Floor(self.floor_colours[self.goal_up]))

        '''
        +FLOORS:
        .......
        XXXXXXX
        XF..X.X
        X.....X
        XF..X.X
        XXXXXXX
        .......
        '''
        # Goal + terminal
        if self.goal_up:
            self.grid.set(5,2,Goal())
            self.grid.set(5,4,Fake_Goal())
        else:
            self.grid.set(5,4,Goal())
            self.grid.set(5,2,Fake_Goal())
        '''
        +GOAL:
        .......
        XXXXXXX
        XFFFXGX
        X.....X
        XFFFXGX
        XXXXXXX
        .......
        '''

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Choose left or right based on starting information"
    
    def reset(self, *, seed = None, options = None,):
        self.task_count += 1
        if self.task_count >= self.task_switch: # Switch task after task_switch episodes
            self.task_count = 0
            self.goal_up = not self.goal_up # Swaps between goal up and goal down
        return super().reset()
        

class Fake_Goal(WorldObj):
    def __init__(self):
        super().__init__("lava", 'green')

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color]) # Same as goal

def main():
    env = T_Maze(render_mode="human")
    
    env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
    state = env.reset(seed=42)
    print("State: ", state)

    state = env.step(0)

    print("Next state: ", state)
    '''
    print(type(state[0]))
    state_dict = state[0]
    print("State type: ", type(state_dict["image"]))
    print("State direction: ", state_dict["direction"])

    print(env.observation_space.spaces["image"].shape)
    '''
    # enable manual control for testing
    '''
    manual_control = ManualControl(env, seed=42)
    manual_control.start()'
    '''

if __name__=="__main__":
    main()