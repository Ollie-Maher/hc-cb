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
        size=5,
        agent_start_pos=(1, 2),
        agent_start_dir=0,
        max_steps: int | None = 100,
        task_switch: int = 1,
        agent_view_size: int = 3,
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
            agent_view_size=agent_view_size,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Choose left or right based on starting information"
    
    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        EDITED: Return full reward
        """

        return 1 #- 0.9 * (self.step_count / self.max_steps)

    def _gen_grid(self, width, height):
        # Gen grid
        self.grid = Grid(width, height)
        # Walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.set(2,1,Wall())
        self.grid.set(2,3,Wall())
        '''
        WALLS:
        XXXXX
        X.X.X
        X...X
        X.X.X
        XXXXX
        '''
        # Coloured walls
        self.grid.set(1,1,Colour_Wall(self.floor_colours[self.goal_up]))
        self.grid.set(1,3,Colour_Wall(self.floor_colours[self.goal_up]))

        '''
        +COLOURED WALLS:
        XXXXX
        XCX.X
        X...X
        XCX.X
        XXXXX
        .......
        '''
        # Goal + terminal
        if self.goal_up:
            self.grid.set(3,1,Goal(self.floor_colours[1]))
            self.grid.set(3,3,Fake_Goal(self.floor_colours[0]))
        else:
            self.grid.set(3,3,Goal(self.floor_colours[0]))
            self.grid.set(3,1,Fake_Goal(self.floor_colours[1]))
        '''
        +GOAL:
        XXXXX
        XCXGX
        X...X
        XCXGX
        XXXXX
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

class easy_T_Maze(MiniGridEnv):
    def __init__(
        self,
        size=5,
        agent_start_pos=(1, 2),
        agent_start_dir=0,
        max_steps: int | None = 100,
        task_switch: int = 1,
        agent_view_size: int = 3,
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
            agent_view_size=agent_view_size,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Choose left or right based on starting information"
    
    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        EDITED: Return full reward
        """

        return 1 #- 0.9 * (self.step_count / self.max_steps)

    def _gen_grid(self, width, height):
        # Gen grid
        self.grid = Grid(width, height)
        # Walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.set(2,1,Wall())
        self.grid.set(2,3,Wall())
        '''
        WALLS:
        XXXXX
        X.X.X
        X...X
        X.X.X
        XXXXX
        '''
        # Coloured walls
        self.grid.set(1,1,Colour_Wall(self.floor_colours[self.goal_up]))
        self.grid.set(1,3,Colour_Wall(self.floor_colours[self.goal_up]))

        '''
        +COLOURED WALLS:
        XXXXX
        XCX.X
        X...X
        XCX.X
        XXXXX
        .......
        '''
        # Goal + terminal
        if self.goal_up:
            self.grid.set(3,1,Goal())
            self.grid.set(3,3,Colour_Wall(self.floor_colours[self.goal_up]))
        else:
            self.grid.set(3,3,Goal())
            self.grid.set(3,1,Colour_Wall(self.floor_colours[self.goal_up]))
        '''
        +GOAL:
        XXXXX
        XCXGX
        X...X
        XCXGX
        XXXXX
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

class water_maze(MiniGridEnv):
    def __init__(
        self,
        size=7,
        agent_start_dir=0,
        max_steps: int | None = 100,
        task_switch: int = 1,
        agent_view_size: int = 3,
        **kwargs,
    ):
        # Might need to implement and override for the agent start pos
        self.task_position = 0
        self.task_positions = [(5,5),(1,5),(5,1)]
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        self.task_switch = task_switch
        self.task_count = 0

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Find the hidden goal"
    
    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        EDITED: Return full reward
        """

        return 1 #- 0.9 * (self.step_count / self.max_steps)

    def _gen_grid(self, width, height):
        # Gen grid
        self.grid = Grid(width, height)
        # Walls
        self.grid.wall_rect(0, 0, width, height)
        '''
        WALLS:
        XX...XX
        X.....X
        .......
        .......
        .......
        X.....X
        XX...XX
        '''
        # Coloured walls
        for i in range(3):
            self.grid.set(0,i+2,Colour_Wall('blue'))
            self.grid.set(6,i+2,Colour_Wall('red'))
            self.grid.set(i+2,0,Colour_Wall('yellow'))
            self.grid.set(i+2,6,Colour_Wall('green'))
        '''
        COLOURED WALLS:
        XXYYYXX
        X.....X
        B.....R
        B.....R
        B.....R
        X.....X
        XXGGGXX
        '''
        # Invisible goal
        self.grid.set(2,2,Invisible_Goal())
        '''
        COLOURED WALLS:
        XXYYYXX
        X.....X
        B.O...R
        B.....R
        B.....R
        X.....X
        XXGGGXX
        '''
        # Place the agent
        self.agent_pos = self.task_positions[self.task_position]
        self.agent_dir = self.agent_start_dir
        
    def reset(self, *, seed = None, options = None,):
        self.task_count += 1
        if self.task_count >= self.task_switch: # Switch task after task_switch episodes
            self.task_count = 0
            
            self.task_position += 1
            if self.task_position >= len(self.task_positions):
                self.task_position = 0
            
        return super().reset()

    
class Fake_Goal(WorldObj):
    def __init__(self, color='green'):
        super().__init__("lava", color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color]) # Same as goal

class Goal(WorldObj):
    def __init__(self, color='green'):
        super().__init__("goal", color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color]) # Same as goal

class Invisible_Goal(WorldObj):
    def __init__(self):
        super().__init__("goal", 'green')

    def can_overlap(self):
        return True

    def render(self, img):
        pass

class Colour_Wall(WorldObj):
    def __init__(self, color):
        super().__init__("wall", color)

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
    
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

if __name__=="__main__":
    main()