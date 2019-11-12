import numpy as np
from LeducHoldem import Game
import copy
import queue
import utils
from utils import RegretSolver, exploitability, generateOutcome, RegretSolverPlus, simulate
import time

for bidmaximum in range(4,6):
	for i in range(10,20):
		savepath="games/leduc_3_"+str(bidmaximum)+"_"+str(i)
		print(bidmaximum, i, savepath)
		game=Game(bidmaximum=bidmaximum, savepath=savepath, save=True)