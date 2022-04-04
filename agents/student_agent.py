# Student agent: Add your own agent here
import numpy as np

from agents.agent import Agent
from store import register_agent
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def check_valid_step(self, start_pos, end_pos, adv_pos, chess_board, max_step):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is boarder
        """
        r, c = end_pos
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True
        """

        x_diff = abs(start_pos[0] - end_pos[0])
        y_diff = abs(start_pos[1] - end_pos[1])
        if (x_diff + y_diff <= max_step):
            return True
        else:
            return False

    def dist(self, c1, c2):
        x_dist = c1[0] - c2[0]
        y_dist = c1[1] - c2[1]
        return (x_dist, y_dist)

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        poss_moves = []
        x_dist = my_pos[0] - adv_pos[0]
        y_dist = my_pos[1] - adv_pos[1]
        size = len(chess_board)
        best_moves = []
        if (not chess_board[adv_pos[0],adv_pos[1], 0]):
            best_moves.append((adv_pos[0]-1,adv_pos[1]))
        if (not chess_board[adv_pos[0], adv_pos[1], 1]):
            best_moves.append((adv_pos[0], adv_pos[1]+1))
        if (not chess_board[adv_pos[0], adv_pos[1], 2]):
            best_moves.append((adv_pos[0]+1, adv_pos[1]))
        if (not chess_board[adv_pos[0], adv_pos[1], 3]):
            best_moves.append((adv_pos[0], adv_pos[1]-1))
        for i in range(size):
            for j in range(size):
                #dir = np.random.randint(0, 4)
                coord = (i,j)
                #print("hi")
                #print(coord)
                #print(dir)
                if (self.check_valid_step(my_pos, coord,  adv_pos, chess_board, max_step) and coord != adv_pos):
                    #print(coord)
                    poss_moves.append(coord)
        min = 100
        final = 0
        print(best_moves)
        for move in poss_moves:
            #print(move)
            for move2 in best_moves:
                x_d = move[0] - move2[0]
                y_d = move[1] - move2[1]
                if (abs(x_d)+abs(y_d) < min):
                    min = abs(x_d)+abs(y_d)
                    final = move
        dis = self.dist(final, adv_pos)
        if (abs(dis[0]) <= abs(dis[1])):
            if (dis[1] < 0):
                dir = 1
            else:
                dir = 3
        else:
            if (dis[0] < 0):
                dir = 2
            else:
                dir = 0



        return final, dir
