# Student agent: Add your own agent here
import math
from copy import deepcopy

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
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

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
        return is_reached

    def dist(self, c1, c2):
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


    def bsf2(self, chess_board, start_pos1, adv_pos1, list, final):
        start_pos = np.asarray(start_pos1)
        adv_pos = np.asarray(adv_pos1)
        # BFS
        if tuple(start_pos) in list:
            return tuple(start_pos), 0
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False

        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue
                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if tuple(next_pos) in list and tuple(next_pos) not in final:
                    is_reached = True
                    break
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

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


    def pick_moves(self, chess_board, adv_pos, poss_moves, best_moves):
        #BFS
        min = 100
        final = []
        for move in best_moves:
            bfs = self.bsf2(chess_board, move, adv_pos, poss_moves, final)
            if bfs == False:
                continue
            else:
                if bfs[0] not in final:
                    final.append(bfs[0])
                if (bfs[1] <= min):
                    min = bfs[1]
                    min_m = bfs[0]
        #if BFS fails -> Backup
        if final == []:
            for move in poss_moves:
                for move2 in best_moves:
                    x_d = move[0] - move2[0]
                    y_d = move[1] - move2[1]
                    if (abs(x_d) + abs(y_d) <= min):
                        min = abs(x_d) + abs(y_d)
                        min_m = move
                        final.append(move)
        return final, min_m

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

    def set_barrier(self, r, c, dir, chess_board):
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def check_endgame(self, chess_board, player_pos, opponent_pos):
        board_size = int(math.sqrt(chess_board.size) / 2)

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

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(player_pos)
        p1_r = find(opponent_pos)
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)

        #print("Point Counting")
        if p0_r == p1_r:
            return  0
        if p0_score > p1_score:
            return  1
        elif p0_score < p1_score:
            return  -1
        else:
            return 0.5



    def minimax(self, chess_board, my_pos, adv_pos, poss_moves, max_step, depth=0):
        dict = {}
        lose = 0
        tie = 0
        for move in poss_moves:
            copy = deepcopy(chess_board)
            opp_utility = []
            self.set_barrier(move[0][0], move[0][1], move[1], copy)
            # we calculate the utility of our move by checking endgame and seeing if we won, lost or game still going
            utility = self.check_endgame(copy, move[0], adv_pos)
            # if utility is 1 then we win so we can break out and perform this move
            if utility == 1 and depth == 0:
                return move
            # if utility is -1 we lose so we eliminate this move as an option and continue
            if utility == -1 and depth == 0:
                lose = move
            if utility == 0.5 and depth == 0:
                tie = move
            # if utility is 0 the result is undetermined so we keep this move as an option and continue
            if utility == 0 and depth == 0:
                # we run the opponents move in response to each of our potential moves
                best_moves = self.find_best_moves(copy, move[0])
                poss_moves = self.get_valid_moves(copy, adv_pos, move[0], max_step)
                moves = self.pick_moves(copy, move[0], poss_moves, best_moves)
                final = self.generate_full_moves(copy, moves[0])
                # we calculate the opponents utility for all their possible moves and return min utility
                opp_utility = min(self.minimax(copy, adv_pos, move[0], final, max_step, 1))
                dict[move] = opp_utility*-1
            if depth == 1:
                opp_utility.append(utility)
        if depth == 1:
            return opp_utility
        if bool(dict):
            max_move = max(dict, key=dict.get)
        else:
            if tie != 0:
                return tie
            else:
                return lose
        if dict[max_move] == 0:
            m = [k for k, v in dict.items() if v == 0]
            return m
        return max_move


        # return the best move
    def generate_full_moves(self, chess_board, moves):
        final = []
        for move in moves:
            for i in range(4):
                r, c = move
                if not chess_board[r, c, i]:
                    final.append((move, i))
        return final

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
        moves = self.pick_moves(chess_board,adv_pos,poss_moves,best_moves)

        final = self.generate_full_moves(chess_board, moves[0])

        result = self.minimax(chess_board,my_pos,adv_pos,final, max_step)
        dir = -1
        if type(result) == list:
            for move in result:
                if (moves[1] == move[0]):
                    dir = self.choose_dir(adv_pos, chess_board, move[0])
                if dir != -1:
                    return move[0], dir
            if dir == -1:
                return result[0][0], result[0][1]

        return result[0], result[1]
