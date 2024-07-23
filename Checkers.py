import tkinter as tk
from tkinter import messagebox
import random

Board = list[list[int]]
Position = tuple[int, int]
Positions = list[Position]
Moves = list[tuple[Position, Positions]]


class Checkers(object):
    """
    checkers class contains methods to play checkers
    """

    WHITE = 1
    WHITE_MAN = 1
    WHITE_KING = 3
    BLACK = 0
    BLACK_MAN = 2
    BLACK_KING = 4
    DX = [1, 1, -1, -1]
    DY = [1, -1, 1, -1]
    OO = 10 ** 9

    def __init__(self, size: int = 8) -> None:
        """Make the initial board of the game

        Args:
            size (int, optional): size of the checkers board. Defaluts to 8.
        Raises:
            Exception: if the size is not even or less than 4
        """
        if size % 2 != 0 or size < 4:
            raise Exception("The size of the board must be even and graeter than 3")

        self.size = size
        self.board = []
        piece = self.WHITE_MAN
        for i in range(size):
            l = []
            f = i % 2 == 1
            if i == size / 2 - 1:
                piece = 0
            elif i == size / 2 + 1:
                piece = self.BLACK_MAN
            for _ in range(size):
                if f:
                    l.append(piece)
                else:
                    l.append(0)
                f = not f
            self.board.append(l)

        self.stateCounter = {}

    def printBoard(self, x: int = None, y: int = None):
        """Print the game board in stdout, the given position is printed in green

        Args:
            x (int, optional): the new x position of the recently played move. Defaults to None.
            y (int, optional): the new y position of the recently played move. Defaults to None.
        """
        for i in range(self.size):
            for j in range(self.size):
                if i == x and j == y:
                    print("\033[92m", end="")

                if self.board[i][j] == 0:
                    print("-", end=" ")
                else:
                    print(self.board[i][j], end=" ")

                if i == x and j == y:
                    print("\033[0m", end="")
            print()
    def encodeBoard(self) -> int:
        """Encode the game board so that each state can be represented by a single unique integer

        Returns:
            int: the value of the encoded game board
        """
        value = 0
        for i in range(self.size):
            for j in range(self.size):
                # make the minimum value = 5, 
                # so that it's greater than the greatest value of the board (4)
                num = i * self.size + j + 5
                value += num * self.board[i][j]


        encoded_board = value
        if encoded_board not in self.stateCounter:
            self.stateCounter[encoded_board] = 0
        
        return encoded_board


    def getBoard(self):
        """Get Game board"""

        return [row[:] for row in self.board]

    def setBoard(self, board):
        """Set game board"""

        if len(board) != len(self.board) or len(board[0]) != len(self.board[0]):
            raise ValueError("Dimensions of the incoming board do not match the current board")


        for i in range(len(board)):
            for j in range(len(board[0])):
                self.board[i][j] = board[i][j]

    def isValid(self, x: int, y: int) -> bool:
        """Check if the given position is inside the board

        Args:
            x (int): x position
            y (int): y position

        Returns:
            bool: the given position is valid
        """
        return x >= 0 and x < self.size and y >= 0 and y < self.size

    def nextPositions(self, x: int, y: int) -> tuple[Positions, Positions]:
        """Get the possible next positions for a given position

        Args:
            x (int): x position
            y (int): y position

        Returns:
            (Positions, Positions): next normal positions, next capture positions
        """
        if self.board[x][y] == 0:
            return []

        player = self.board[x][y] % 2
        captureMoves = []
        normalMoves = []
        sign = 1 if player == self.WHITE else -1
        # only forward for men and both forward and backward for Kings
        rng = 2 if self.board[x][y] <= 2 else 4
        for i in range(rng):
            nx = x + sign * self.DX[i]
            ny = y + sign * self.DY[i]
            if self.isValid(nx, ny):
                if self.board[nx][ny] == 0:
                    normalMoves.append((nx, ny))
                elif self.board[nx][ny] % 2 == 1 - player:
                    nx += sign * self.DX[i]
                    ny += sign * self.DY[i]
                    if self.isValid(nx, ny) and self.board[nx][ny] == 0:
                        captureMoves.append((nx, ny))

        return normalMoves, captureMoves

    def nextMoves(self, player: int) -> Moves:
        """Get the next moves of the game board for a certian player

        Args:
            player (int): the type of player (WHITE, BLACK)

        Returns:
            Moves: valid moves for the player
        """
        captureMoves = []
        normalMoves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] != 0 and self.board[x][y] % 2 == player:
                    normal, capture = self.nextPositions(x, y)
                    if len(normal) != 0:
                        normalMoves.append(((x, y), normal))
                    if len(capture) != 0:
                        captureMoves.append(((x, y), capture))
        if len(captureMoves) != 0:
            return captureMoves
        return normalMoves

    def playMove(self, x: int, y: int, nx: int, ny: int) -> tuple[bool, int, bool]:
        """Change the board by playing a move from (x, y) to (nx, ny)

        Args:
            x (int): the old x position
            y (int): the old y position
            nx (int): the new x position
            ny (int): the new y position

        Returns:
            canCapture (bool): if the player can capture more pieces.  
            removed (int): the removed piece (if any).  
            promoted (bool) if the current piece is promoted).  
        """
        self.board[nx][ny] = self.board[x][y]
        self.board[x][y] = 0

        removed = 0
        if abs(nx - x) == 2:  # capture move
            dx = nx - x
            dy = ny - y
            removed = self.board[x + dx // 2][y + dy // 2]
            self.board[x + dx // 2][y + dy // 2] = 0  # remove captured piece

        # promote to king
        if self.board[nx][ny] == self.WHITE_MAN and nx == self.size - 1:
            self.board[nx][ny] = self.WHITE_KING
            return False, removed, True
        if self.board[nx][ny] == self.BLACK_MAN and nx == 0:
            self.board[nx][ny] = self.BLACK_KING
            return False, removed, True

        if abs(nx - x) != 2:
            return False, removed, False

        return True, removed, False

    def undoMove(self, x: int, y: int, nx: int, ny: int, removed=0, promoted=False):
        """Undo a move and return the board to its previous state

        Args:
            x (int): the old x position of the played move
            y (int): the old y position of the played move
            nx (int): the new x position of the played move
            ny (int): the new y position of the played move
            removed (int, optional): the removed piece (if any). Defaults to 0.
            promoted (bool, optional): if the played piece was recently promoted. Defaults to False.
        """
        if promoted:
            if self.board[nx][ny] == self.WHITE_KING:
                self.board[nx][ny] = self.WHITE_MAN

            if self.board[nx][ny] == self.BLACK_KING:
                self.board[nx][ny] = self.BLACK_MAN

        self.board[x][y] = self.board[nx][ny]
        self.board[nx][ny] = 0

        if abs(nx - x) == 2:
            dx = nx - x
            dy = ny - y
            self.board[x + dx // 2][y + dy // 2] = removed

    def randomPlay(
        self, player: int, moves: Moves = None, enablePrint=True
    ) -> tuple[bool, bool]:
        """play a random play for a given player, 
        if the player should continue capturing, then it will

        Args:
            board (Board): the game board
            player (int): the type of the player (WHITE, BLACK)
            moves (Moves, optional): the next moves 
                (used in case of continuing capturing). Defaults to None
            enablePrint (bool, optional): when true the function prints 
                the board after playing. Defaults to True

        Returns:
            continue (bool): false if there is no further plays.  
            reset (bool): true when there is a captured piece, 
                used to reset the counter of the draw condition.  
        """
        if moves == None:
            moves = self.nextMoves(player)
        if len(moves) == 0:
            if enablePrint:
                print(("WHITE" if player == self.BLACK else "BLACK") + " Player wins")
            return False, False
        randomMove = random.choice(moves)
        x, y = randomMove[0]
        nx, ny = random.choice(randomMove[1])

        if enablePrint:
            print(f"Move from ({x}, {y}) to ({nx}, {ny})")
        canCapture, removed, _ = self.playMove(x, y, nx, ny)
        if enablePrint:
            self.printBoard(nx, ny)

        if canCapture:
            _, nextCaptures = self.nextPositions(nx, ny)
            if len(nextCaptures) != 0:
                self.randomPlay(
                    player, [((nx, ny), nextCaptures)], enablePrint=enablePrint
                )

        reset = removed != 0
        return True, reset

    def evaluate1(self, maximizer: int) -> int:
        """evaluate the current state of the board

        Args:
            maximizer (int): the type of the maximizer player (WHITE, BLACK)

        Returns:
            int: score of the board
        """
        # score = (2*maximizer_kings+maximizer_men - (2*opponent_kings + opponent_men))*1000
        
        score = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != 0:
                    if self.board[i][j] % 2 == maximizer:
                        score += (self.board[i][j] + 1) // 2
                    else:
                        score -= (self.board[i][j] + 1) // 2
        return score * 1000

    def cellContains(self, x: int, y: int, player: int) -> bool:
        """return if cell at (x, y) contains player

        Args:
            x (int): x position of cell
            y (int): y position of cell
            player (int): type of player (WHITE/BLACK)

        Returns:
            bool: if cell at (x, y) contains player
        """
        return self.board[x][y] != 0 and self.board[x][y] % 2 == player

    def endGame(self, maximizer: int) -> int:
        """evaluate the current state of the board based on end game strategies
            between maximizer player and the opponent

        Args:
            maximizer (int): the type of the maximizer player (WHITE, BLACK)

        Returns:
            int: score of the board
        """
        score1 = 0
        score2 = 0
        maxPieces = 0
        minPieces = 0
        rowScore = 0
        base = 0 if maximizer == self.WHITE else self.size-1
        minimizer = 1 - maximizer
        minimizerPositions = []
        for x in range(self.size):
            for y in range(self.size):
                if self.cellContains(x, y, minimizer):
                    minimizerPositions.append((x, y))

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != 0:
                    if self.board[i][j] % 2 == maximizer:
                        maxPieces += 1
                        if (self.board[i][j] + 1) // 2 == 1:
                            rowScore += abs(base-i)
                        score1 += (self.board[i][j] + 1) // 2
                        for x,y in minimizerPositions:
                            score2 += (x-i)**2 + (y-j)**2
                    else:
                        minPieces += 1
                        score1 -= (self.board[i][j] + 1) // 2

        # penalize if the minimizer is in the corner to be able to trap him at the end of the game                   
        minimizerCorner = 0
        for x, y in minimizerPositions:
            if (x,y) == (0, 1) or (x,y) == (1, 0) or (x, y) == (self.size-1, self.size-2) \
                or (x,y) == (self.size-2, self.size-1):
                minimizerCorner = 1

        maximizerCorner = 0
        if self.cellContains(0, 1, maximizer) or self.cellContains(1, 0, maximizer) \
            or self.cellContains(self.size-1, self.size-2, maximizer) \
            or self.cellContains(self.size-2, self.size-1, maximizer):
            maximizerCorner = 1

        if maxPieces > minPieces:   #come closer to opponent
            return score1*1000 - score2 - minimizerCorner*5 + rowScore*10
        else:    # run away
            return score1*1000 + score2 + maximizerCorner*5

    def evaluate2(self, maximizer: int) -> int:
        """evaluate the current state of the board

        Args:
            maximizer (int): the type of the maximizer player (WHITE, BLACK)

        Returns:
            int: score of the board
        """
        
        men = 0
        kings = 0
        backRow = 0
        middleBox = 0
        middleRow = 0
        vulnerable = 0
        protected = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != 0:
                    sign = 1 if self.board[i][j] % 2 == maximizer else -1
                    if self.board[i][j] <= 2:
                        men += sign*1
                    else:
                        kings += sign*1
                    if sign == 1 and ((i == 0 and maximizer == self.WHITE) or (i == self.size-1 and maximizer == self.BLACK)):
                        backRow += 1
                    if i == self.size/2-1 or i == self.size/2:
                        if j >= self.size/2-2 and j < self.size/2+2:
                            middleBox += sign*1
                        else:
                            middleRow += sign*1

                    myDir = 1 if maximizer == self.WHITE else -1
                    vul = False
                    for k in range(4):
                        x = i + self.DX[k]
                        y = j + self.DY[k]
                        n = i - self.DX[k]
                        m = j - self.DY[k]
                        opDir = abs(x-n)/(x-n)
                        if self.isValid(x, y) and self.board[x][y] != 0 and self.board[x][y] % 2 != maximizer \
                            and self.isValid(n, m) and self.board[n][m] == 0 and (self.board[x][y] > 2 or myDir != opDir):
                            vul = True
                            break
                    
                    if vul:
                        vulnerable += sign*1
                    else:
                        protected += sign*1
                
        return men*2000 + kings*4000 + backRow*400 + middleBox*250 + middleRow*50 - 300*vulnerable + 300*protected

    def stateValue(self, maximizer: int) -> int:
        """Get the value of the board state,
        when the maximizer's pieces are greater than the minimizer's, 
        penalize repeating the same state

        Args:
            maximizer (int): the type of the maximizer player (WHITE/BLACK)

        Returns:
            int: value of the board state
        """
        maxPieces = 0
        minPieces = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != 0:
                    if self.board[i][j] % 2 == maximizer:
                        maxPieces += 1
                    else:
                        minPieces += 1
        if maxPieces > minPieces:
            return -self.stateCounter[self.encodeBoard()]
        return 0

    def minimax(
        self,
        player,
        maximizer,
        depth=0,
        alpha=-OO,
        beta=OO,
        maxDepth=4,
        evaluate=None,
        moves=None
    ) -> int:
        """Get the score of the board using alpha-beta algorithm

        Args:
            player (int): the type of the current player (WHITE, BLACK)
            maximizer (int): the type of the maximizer player (WHITE, BLACK)
            depth (int, optional): the current depth of the algorithm. Defaults to 0.
            alpha (int, optional): the value of alpha. Defaults to -OO.
            beta (int, optional): the value of beta of the algorithm. Defaults to OO.
            maxDepth (int, optional): the higher the max depth, 
                the harder the level of th play and the more time the algorithm will take. Defaults to 4.
            evaluate (Callable[[int], int], optional): evaluation function. Defaults to evaluate2
            moves (Moves, optional): the next capture moves (if any). Defaults to None.

        Returns:
            int|float : score of the baord
        """
        if moves == None:
            moves = self.nextMoves(player)
        if len(moves) == 0 or depth == maxDepth:
            score = evaluate(self, maximizer)
            # if there is no escape from losing, maximize number of moves to lose
            if score < 0:
                score += depth
            return score

        bestValue = -self.OO
        if player != maximizer:
            bestValue = self.OO

        # sort moves by the minimum next positions
        moves.sort(key=lambda move: len(move[1]))
        for position in moves:
            x, y = position[0]
            for nx, ny in position[1]:

                canCapture, removed, promoted = self.playMove(x, y, nx, ny)
                played = False

                if canCapture:
                    _, nextCaptures = self.nextPositions(nx, ny)
                    if len(nextCaptures) != 0:
                        played = True
                        nMoves = [((nx, ny), nextCaptures)]
                        if player == maximizer:
                            bestValue = max(
                                bestValue,
                                self.minimax(player, maximizer, depth + 1, alpha, beta, maxDepth, evaluate, nMoves)
                            )
                            alpha = max(alpha, bestValue)
                        else:
                            bestValue = min(
                                bestValue,
                                self.minimax(player, maximizer, depth + 1, alpha, beta, maxDepth, evaluate, nMoves)
                            )
                            beta = min(beta, bestValue)

                if not played:
                    if player == maximizer:
                        bestValue = max(
                            bestValue,
                            self.minimax(1 - player, maximizer, depth + 1, alpha, beta, maxDepth, evaluate)
                        )
                        alpha = max(alpha, bestValue)
                    else:
                        bestValue = min(
                            bestValue,
                            self.minimax(1 - player, maximizer, depth + 1, alpha, beta, maxDepth, evaluate)
                        )
                        beta = min(beta, bestValue)

                self.undoMove(x, y, nx, ny, removed, promoted)

                if beta <= alpha:
                    break
            if beta <= alpha:
                break

        return bestValue

    def minimaxPlay(
        self,
        player,
        moves=None,
        maxDepth=4,
        evaluate=evaluate2,
        enablePrint=True
    ) -> tuple:
        """play a move using minimax algorithm
            if the player should continue capturing, it will

        Args:
            player (int): the type of the player (WHITE, BLACK)
            moves (Moves, optional): the next capture moves (if any). Defaults to None.
            maxDepth (int, optional): the max depth of the minimax algorithm
                the higher the max depth, the harder the level of th play 
                and the more time the algorithm will take. Defaults to 4.
            enablePrint (bool, optional): if true it prints the game board 
                to stdout after playing the move. Defaults to True.

        Returns:
            continue (bool): false if there is no further plays.  
            reset (bool): true when there is a captured piece, 
                used to reset the counter of the draw condition.
        """

        if moves == None:
            moves = self.nextMoves(player)
        if len(moves) == 0:
            if enablePrint:
                print(("WHITE" if player == self.BLACK else "BLACK") + " Player wins")
            return False, False

        encoded_board = self.encodeBoard()
        if encoded_board in self.stateCounter:
            self.stateCounter[encoded_board] += 1
        else:
            self.stateCounter[encoded_board] = 1

        random.shuffle(moves)
        bestValue = -self.OO
        bestMove = None

        for position in moves:
            x, y = position[0]
            for nx, ny in position[1]:
                _, removed, promoted = self.playMove(x, y, nx, ny)
                value = self.minimax(1 - player, player, maxDepth=maxDepth, evaluate=evaluate)
                value += 2*self.stateValue(player)  
                self.undoMove(x, y, nx, ny, removed, promoted)
                if value > bestValue:
                    bestValue = value
                    bestMove = (x, y, nx, ny)

        x, y, nx, ny = bestMove
        if enablePrint:
            print(f"Move from ({x}, {y}) to ({nx}, {ny})")
        canCapture, removed, _ = self.playMove(x, y, nx, ny)
        if enablePrint:
            self.printBoard(nx, ny)

        if canCapture:
            _, captures = self.nextPositions(nx, ny)
            if len(captures) != 0:
                self.minimaxPlay(player, [((nx, ny), captures)], maxDepth, evaluate, enablePrint)

        encoded_board = self.encodeBoard()
        if encoded_board in self.stateCounter:
            self.stateCounter[encoded_board] += 1
        else:
            self.stateCounter[encoded_board] = 1
        reset = removed != 0
        return True, reset


window = tk.Tk()
window.geometry(
"700x680+400+100"
)  
window.title("Checkers")
IMG_SIZE = 60

black_man_img = tk.PhotoImage(file='assets/black_man.png')
black_man_img = black_man_img.subsample(IMG_SIZE // 10, IMG_SIZE // 10)

black_king_img = tk.PhotoImage(file='assets/black_king.png')
black_king_img = black_king_img.subsample(IMG_SIZE // 10, IMG_SIZE // 10)

white_man_img = tk.PhotoImage(file='assets/white_man.png')
white_man_img = white_man_img.subsample(IMG_SIZE // 10, IMG_SIZE // 10)

white_king_img = tk.PhotoImage(file='assets/white_king.png')
white_king_img = white_king_img.subsample(IMG_SIZE // 10, IMG_SIZE // 10)

blank_img = tk.PhotoImage(file='assets/blank.png')
blank_img = blank_img.subsample(IMG_SIZE // 10, IMG_SIZE // 10)


# class Coin(Enum):
#     HEADS = 0
#     TAILS = 1
# class Mode(Enum):
#     SINGLE_PLAYER = 0
#     MULTIPLE_PLAYER = 1
# class Algorithm(Enum):
#     MINIMAX = 0
#     RANDOM = 1

SINGLE_PLAYER = 0
MINIMAX = 0
RANDOM = 1
HEADS = 0
TAILS = 0
CHECKER_SIZE = 8
GAME_MODE = SINGLE_PLAYER
#STARTING_PLAYER = Checkers.BLACK
USED_ALGORITHM = MINIMAX
MAX_DEPTH = 5
EVALUATION_FUNCTION = Checkers.evaluate2
INCREASE_DEPTH = True

def from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'

class GUI:
    def __init__(self, who_play) -> None:
        super().__init__()
        self.game = Checkers(CHECKER_SIZE)
        self.history = [self.game.getBoard()]
        self.historyPtr = 0

        self.algo = USED_ALGORITHM
        self.maxDepth = MAX_DEPTH

        self.difficulty = "Hard"


        self.player = who_play

        if self.player == Checkers.WHITE and GAME_MODE == SINGLE_PLAYER:
            if self.algo == MINIMAX:
                self.game.minimaxPlay(1-self.player, maxDepth=self.maxDepth, evaluate=EVALUATION_FUNCTION, enablePrint=False)
            elif self.algo == RANDOM:
                self.game.randomPlay(1-self.player, enablePrint=False)
            self.history = [self.game.getBoard()]
        
        self.lastX = None
        self.lastY = None
        self.willCapture = False
        self.cnt = 0
        self.btn = [[None]*self.game.size for _ in range(self.game.size)]

        frm_board = tk.Frame(master=window)
        frm_board.pack(fill=tk.BOTH, expand=True)
        for i in range(self.game.size):
            frm_board.columnconfigure(i, weight=1, minsize=IMG_SIZE)
            frm_board.rowconfigure(i, weight=1, minsize=IMG_SIZE)

            for j in range(self.game.size):
                frame = tk.Frame(master=frm_board)
                frame.grid(row=i, column=j, sticky="nsew")

                self.btn[i][j] = tk.Button(master=frame, width=IMG_SIZE, height=IMG_SIZE, relief=tk.FLAT)
                self.btn[i][j].bind("<Button-1>", self.click)
                self.btn[i][j].pack(expand=True, fill=tk.BOTH)
                

        frm_options = tk.Frame(master=window)
        frm_options.pack(expand=True)

        btn_rules = tk.Button(master=frm_options, command=self.show_rules, text="Rules")
        btn_rules.pack(side=tk.LEFT, padx=5, pady=5)

        self.difficulty_var = tk.StringVar(window)
        self.difficulty_var.set(self.difficulty) 
        self.difficulty_menu = tk.OptionMenu(frm_options, self.difficulty_var, "Easy", "Hard", command=self.change_difficulty)
        self.difficulty_menu.pack(side=tk.LEFT, padx=5, pady=5)

        btn_undo = tk.Button(master=frm_options, command=self.undo, text="Undo")
        btn_undo.pack(side=tk.LEFT, padx=5, pady=5)

        btn_redo = tk.Button(master=frm_options, command=self.redo, text="Redo")
        btn_redo.pack(side=tk.LEFT, padx=5, pady=5)

        frm_counter = tk.Frame(master=window)
        frm_counter.pack(expand=True)
        self.lbl_counter = tk.Label(master=frm_counter)
        self.lbl_counter.pack()

        user_color = "Black" if who_play == Checkers.BLACK else "White"
        computer_color = "White" if who_play == Checkers.BLACK else "Black"

        user_label = tk.Label(master=window, text=f"User: {user_color}", foreground="blue", font=("Arial", 12, "bold"))
        user_label.pack()
        computer_label = tk.Label(master=window, text=f"Computer: {computer_color}", foreground="red", font=("Arial", 12, "bold"))
        computer_label.pack()
        
        self.update()
        nextPositions = [move[0] for move in self.game.nextMoves(self.player)]
        self.highlight(nextPositions)
        window.mainloop()

    def show_rules(self):
        # Define the rules text
        rules_text = """
        Rules of the game:
        1. The game is played on an 8x8 grid.\n
        2. Each player starts with 12 pieces, placed on the dark squares of the three rows closest to their side.\n
        3. Players alternate turns, with Black making the first move.\n
        4. Pieces move diagonally forward. They can only move one square at a time, except when capturing.\n
        5. A piece can capture an opponent's piece by jumping over it diagonally to an unoccupied square immediately beyond.\n
        6. If a piece reaches the opponent's back row, it becomes a "king" and gains the ability to move and capture diagonally backward.\n
        7. The game ends when one player has no legal moves remaining, or one player captures all of the opponent's pieces.\n
        """
        
        # Create a popup window to display the rules
        rules_window = tk.Toplevel(window)
        rules_window.geometry("500x350+500+200")
        rules_window.title("Rules of the Game")
        
        # Create a label to display the rules text
        lbl_rules = tk.Label(rules_window, text=rules_text, wraplength=400, justify=tk.LEFT)
        lbl_rules.pack(padx=10, pady=10)

    def change_difficulty(self, *args):
        new_difficulty = self.difficulty_var.get()
        
        if new_difficulty != self.difficulty:

            self.difficulty = new_difficulty
            if new_difficulty == "hard":

                self.reset_game(MINIMAX)
            else:
                self.reset_game(RANDOM)

    def reset_game(self, level):

        self.game = Checkers(CHECKER_SIZE)
        self.history = [self.game.getBoard()]
        self.historyPtr = 0
        self.lastAIMove = None

        self.algo = level
        if self.player == Checkers.WHITE and GAME_MODE == SINGLE_PLAYER:
            if self.algo == MINIMAX:
                self.game.minimaxPlay(1-self.player, maxDepth=self.maxDepth, evaluate=EVALUATION_FUNCTION, enablePrint=False)
            elif self.algo == RANDOM:
                self.game.randomPlay(1-self.player, enablePrint=False)
            self.history = [self.game.getBoard()]

        self.lastX = None
        self.lastY = None
        self.willCapture = False
        self.cnt = 0

        self.update()
        nextPositions = [move[0] for move in self.game.nextMoves(self.player)]
        self.highlight(nextPositions)

    def update(self):
        for i in range(self.game.size):
            f = i % 2 == 1
            for j in range(self.game.size):

                if f:
                    self.btn[i][j]['bg'] = 'gray20'
                else:
                    self.btn[i][j]['bg'] = 'red'
                img = blank_img
                if self.game.board[i][j] == Checkers.BLACK_MAN:
                    img = black_man_img
                elif self.game.board[i][j] == Checkers.BLACK_KING:
                    img = black_king_img
                elif self.game.board[i][j] == Checkers.WHITE_MAN:
                    img = white_man_img
                elif self.game.board[i][j] == Checkers.WHITE_KING:
                    img = white_king_img

                self.btn[i][j]["image"] = img
                
                f = not f

        self.lbl_counter['text'] = f'Moves without capture: {self.cnt}'
        window.update()
    
    def highlight(self, positions: Positions):
        for x in range(self.game.size):
            for y in range(self.game.size):
                defaultbg = self.btn[x][y].cget('bg')
                self.btn[x][y].master.config(highlightbackground=defaultbg, highlightthickness=3)

        for position in positions:
            x, y = position
            self.btn[x][y].master.config(highlightbackground="yellow", highlightthickness=3)


    def click(self, event):
        info = event.widget.master.grid_info()
        x, y = info["row"], info["column"]
        if self.lastX == None or self.lastY == None:
            moves = self.game.nextMoves(self.player)
            found = (x, y) in [move[0] for move in moves]
            
            if found:
                self.lastX = x
                self.lastY = y
                normal, capture = self.game.nextPositions(x, y)
                positions = normal if len(capture) == 0 else capture
                self.highlight(positions)
            else:
                print("Invalid position: You must select one of highlighted pieces.")
            return

        normalPositions, capturePositions = self.game.nextPositions(self.lastX, self.lastY)
        positions = normalPositions if (len(capturePositions) == 0) else capturePositions
        if (x,y) not in positions:
            print("invalid move: You must select a valid destination for the piece.")
            if not self.willCapture:
                self.lastX = None
                self.lastY = None
                nextPositions = [move[0] for move in self.game.nextMoves(self.player)]
                self.highlight(nextPositions)
            return

        canCapture, removed, _ = self.game.playMove(self.lastX, self.lastY, x, y)
        self.highlight([])
        self.update()
        self.cnt += 1
        self.lastX = None
        self.lastY = None
        self.willCapture = False

        if removed != 0:
            self.cnt = 0
        if canCapture:
            _, nextCaptures = self.game.nextPositions(x, y)
            if len(nextCaptures) != 0:
                self.willCapture = True
                self.lastX = x
                self.lastY = y
                self.highlight(nextCaptures)
                return

        if GAME_MODE == SINGLE_PLAYER:
            cont, reset = True, False
            if self.algo == MINIMAX:
                evaluate = EVALUATION_FUNCTION
                if self.cnt > 20:
                    evaluate = Checkers.endGame
                    if INCREASE_DEPTH:
                        self.maxDepth = 7
                else:
                    evaluate = Checkers.evaluate2
                    self.maxDepth = MAX_DEPTH
                    
                cont, reset = self.game.minimaxPlay(1-self.player, maxDepth=self.maxDepth, evaluate=evaluate, enablePrint=False)
            elif self.algo == RANDOM:
                cont, reset = self.game.randomPlay(1-self.player, enablePrint=False)
            self.cnt += 1
            if not cont:
                messagebox.showinfo(message="You Won!", title="Checkers")
                window.destroy()
                return
            self.update()
            if reset:
                self.cnt = 0
        else:
            self.player = 1-self.player

        if self.cnt >= 100:
            messagebox.showinfo(message="Draw!", title="Checkers")
            window.destroy()
            return
        
        nextPositions = [move[0] for move in self.game.nextMoves(self.player)]
        self.highlight(nextPositions)
        if len(nextPositions) == 0:
            if GAME_MODE == SINGLE_PLAYER:
                messagebox.showinfo(message="You lost!", title="Checkers")
            else:
                winner = "BLACK" if self.player == Checkers.WHITE else "WHITE"
                messagebox.showinfo(message=f"{winner} Player won!", title="Checkers")
            window.destroy()

        self.history = self.history[:self.historyPtr+1]
        self.history.append(self.game.getBoard())
        self.historyPtr += 1

    def undo(self):
        if self.historyPtr > 0 and not self.willCapture:
            self.historyPtr -= 1
            self.game.setBoard(self.history[self.historyPtr])
            self.update()

            self.lastX = self.lastY = None
            nextPositions = [move[0] for move in self.game.nextMoves(self.player)]
            self.highlight(nextPositions)
        else:
            print("Can't undo")
    
    def redo(self):
        if self.historyPtr < len(self.history)-1 and not self.willCapture:
            self.historyPtr += 1
            self.game.setBoard(self.history[self.historyPtr])
            self.update()

            self.lastX = self.lastY = None
            nextPositions = [move[0] for move in self.game.nextMoves(self.player)]
            self.highlight(nextPositions)
        else:
            print("Can't redo")



def toss_result(user_choice):
    import random
    computer_choice = random.choice([HEADS, TAILS])
    if user_choice == computer_choice:
        tk.messagebox.showinfo(message="You won the toss! You'll play first.\n You have Black", title="Toss Result")
        start_game(Checkers.BLACK)
    else:
        tk.messagebox.showinfo(message="Computer won the toss! It'll play first.\n You have White", title="Toss Result")
        start_game(Checkers.WHITE)

def start_game(who_play):
    tos.destroy()
    window.deiconify() 
    GUI(who_play)


tos = tk.Tk()
tos.geometry("300x150+500+300")
tos.title("Toss")

window.withdraw()

lbl_instruction = tk.Label(tos, text="Choose Heads or Tails:")
lbl_instruction.pack(pady=10)

btn_heads = tk.Button(tos, text="Heads", command=lambda: toss_result(HEADS))
btn_heads.pack(pady=5)

btn_tails = tk.Button(tos, text="Tails", command=lambda: toss_result(TAILS))
btn_tails.pack(pady=5)


tos.mainloop()
