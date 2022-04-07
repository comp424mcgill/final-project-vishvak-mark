# Student agent: Add your own agent here
import math

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
        self.autoplay = True

    def check_valid_step(self, start_pos1, end_pos1, adv_pos1, chess_board, max_step):
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
        # checks if shortest distance is reachable
        # does not check whether the path is reachable when factoring in borders
        x_diff = abs(start_pos1[0] - end_pos1[0])
        y_diff = abs(start_pos1[1] - end_pos1[1])
        if (x_diff + y_diff > max_step):
            return False
        start_pos = np.asarray(start_pos1)
        end_pos = np.asarray(end_pos1)
        adv_pos = np.asarray(adv_pos1)
        if np.array_equal(start_pos, end_pos):
            return True

        #check if path is actually reachable
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            # print("hi")
            # print(state_queue)
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(self.moves):
                # print("hi3")
                # print(dir, move)
                if chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                # print("hi2")
                # print(next_pos)
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return is_reached

    def dist(self, c1, c2):
        # print(type(c1))
        x_dist = c1[0] - c2[0]
        y_dist = c1[1] - c2[1]
        return (x_dist, y_dist)

    def barrier_chooser(self, chess_board, end_pos, guess):
        if (not chess_board[end_pos[0], end_pos[1], guess]):
            return guess
        for i in range(4):
            if (not chess_board[end_pos[0], end_pos[1], i]):
                dir = i
                return dir

    def check_endgame(self, chess_board, player_pos, opponent_pos, poss_moves):

        board_size = math.sqrt(chess_board.size / 2)
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

      # turn 1: [(x,y),(a,b),(c,d)]
   # (x,y) _> calc utlity of our mvoe -> find best move for opp in response -> calculate its utlity -> return to top
   # -> repeat for all moves in list
   # -> take the max utlity of all the 3 moves
        is_true = False
        for move in poss_moves:
            # we calculate the utlity of our move

            if is_true:
                break
            else:
                # we run the opp move and calculate it's utility
                pass

        # return the best move

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = player_pos
        p1_r = opponent_pos
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, 0
        if p0_score > p1_score:
            return True, 1000
        elif p0_score < p1_score:
            return True, -1000
        else:
            player_win = -1  # Tie



    def bfs(self, chess_board, start_pos1, end_pos1, adv_pos1, max_step):
        start_pos = np.asarray(start_pos1)
        end_pos = np.asarray(end_pos1)
        adv_pos = np.asarray(adv_pos1)
        #BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        if np.array_equal(start_pos, end_pos):
            return True, state_queue.pop(0)[1]
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue
                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return is_reached, cur_step+1

    def bsf2(self, chess_board, start_pos1, adv_pos1, max_step, list):
        start_pos = np.asarray(start_pos1)
        adv_pos = np.asarray(adv_pos1)
        # BFS
        if tuple(start_pos) in list:
            return start_pos, 0
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        #if np.array_equal(start_pos, end_pos):
            #return True, state_queue.pop(0)[1]
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue
                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if tuple(next_pos) in list:
                    is_reached = True
                    # print(tuple(next_pos))
                    # print(list)
                    break
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        # print(next_pos)
        if tuple(next_pos) not in list:
            return False
        return tuple(next_pos), cur_step + 1

    def find_best_moves(self, chess_board, adv_pos):
        best_moves = []
        # finds the best tiles to move to
        if (not chess_board[adv_pos[0], adv_pos[1], 0]):
            best_moves.append((adv_pos[0] - 1, adv_pos[1]))
        if (not chess_board[adv_pos[0], adv_pos[1], 1]):
            best_moves.append((adv_pos[0], adv_pos[1] + 1))
        if (not chess_board[adv_pos[0], adv_pos[1], 2]):
            best_moves.append((adv_pos[0] + 1, adv_pos[1]))
        if (not chess_board[adv_pos[0], adv_pos[1], 3]):
            best_moves.append((adv_pos[0], adv_pos[1] - 1))
        return best_moves

    def get_valid_moves(self, chess_board, my_pos, adv_pos, max_step):
        poss_moves = []
        size = len(chess_board)
        # gets all tiles on the board that are reachable and adds to poss_moves
        for i in range(size):
            for j in range(size):
                coord = (i, j)
                if (self.check_valid_step(my_pos, coord, adv_pos, chess_board, max_step) and coord != adv_pos):
                    poss_moves.append(coord)
        return poss_moves


    def pick_moves(self, chess_board, adv_pos, max_step, poss_moves, best_moves):
        #BFS
        min = 100
        final = []
        for move in best_moves:
            bfs = self.bsf2(chess_board, move, adv_pos, max_step, poss_moves)
            if bfs == False:
                continue
            else:
                if (bfs[1] < min):
                    min = bfs[1]
                    final = bfs[0]
        #if BFS fails -> Backup
        if final == []:
            for move in poss_moves:
                for move2 in best_moves:
                    x_d = move[0] - move2[0]
                    y_d = move[1] - move2[1]
                    if (abs(x_d) + abs(y_d) < min):
                        min = abs(x_d) + abs(y_d)
                        final = move
        return final

    def choose_dir(self, adv_pos, chess_board, final):
        dis = self.dist(final, adv_pos)
        if (abs(dis[0]) <= abs(dis[1])):
            if (dis[1] < 0 and not chess_board[final[0], final[1], 1]):
                dir = 1
            elif (dis[1] >= 0 and not chess_board[final[0], final[1], 3]):
                dir = 3
            else:
                if (dis[0] < 0):
                    dir = self.barrier_chooser(chess_board, final, 2)
                else:
                    dir = self.barrier_chooser(chess_board, final, 0)

        else:
            if (dis[0] < 0 and not chess_board[final[0], final[1], 2]):
                dir = 2
            elif (dis[0] >= 0 and not chess_board[final[0], final[1], 0]):
                dir = 0
            else:
                if (dis[1] < 0):
                    dir = self.barrier_chooser(chess_board, final, 1)
                else:
                    dir = self.barrier_chooser(chess_board, final, 3)
        return dir


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

        # finds the tiles right next to the opponent that do not have a barrier
        best_moves = self.find_best_moves(chess_board, adv_pos)
        # gets all tiles on the board that are reachable and adds to poss_moves
        poss_moves = self.get_valid_moves(chess_board,my_pos, adv_pos, max_step)
        # finds move from poss_moves that is closest to the 'best tile'
        final = self.pick_moves(chess_board,adv_pos,max_step,poss_moves,best_moves)

        # picks the best possible direction to put the barrier at
        dir = self.choose_dir(adv_pos, chess_board, final)
        return final, dir
