import numpy as np
from LeducHoldem import Game
import copy
import queue
import utils
from utils import RegretSolver, exploitability, generateOutcome, RegretSolverPlus
import time
import argparse
import ficplay
import cfrpsrl
import mccfroutcome
import fittedqfsp


def run(game, path="result",  solvername = "cfrpsrl", Type="default", resultpath=""):
	
	def solve(gamesolver, reporttime=2, timelim = 3000, minimum=0, mrounds = 10000):
		cumutime = 0
		timestamp = time.time()
		result = []
		expls = []
		rounds = []
		rnds = -1
		while rnds < mrounds:
			rnds += 1
			gamesolver.updateAll()
			if rnds % 30 == 0:
				expl = gamesolver.getExploitability()
				print("solvername", solvername, Type, "game", gamepath, "expl", expl, "rounds", rnds)
				expls.append(expl)
				rounds.append(rnds)
				np.savez( "results/" + resultpath+"_"+solvername+"_"+Type, expls = expls, rounds = rounds)
		return (expls,  rounds)

	solver = None
	if solvername == "cfrpsrl":
		solver = cfrpsrl.CFRPSRL(game, Type=Type)
	if solvername == "ficplay":
		solver = ficplay.FICPLAY(game)
	if solvername == "mccfros":
		solver = mccfroutcome.MCCFR_OS(game)
	if solvername == "qfsp":
		solver = fittedqfsp.FICPLAY(game)

	res = solve(solver)
	return res

#mc_rmplus, lazy_rmplus, vanilla_rmplus = run(game, Type="regretmatching+")
#mc_rm, lazy_rm, vanilla_rm = run(game, Type="regretmatching")
"""
algo = "cfrpsrl"
Type = "avg"
print("haha")
bedm = 5
gameid = 1

gamepath="games/leduc_3_"+str(bidm)+"_"+str(gameid) 
game = Game(path=gamepath+ ".npz")

print("game info:", game.numHists, game.numIsets, algo, Type)
res = run(game, solvername=algo, Type=Type)
"""
parser = argparse.ArgumentParser()
algo = None
Type = "default"
game = None
betm = 4
cards = 3
gameid = 0
parser.add_argument("--algo", help="specify an algorithm: cfrpsrl, mccfros or fictplay", type=str)
parser.add_argument("--type", help="specify the exploration strategy: if algo == cfrpsrl, you can select default, ordinary or random", type=str)
parser.add_argument("--cards", help="specify the exploration strategy", type=int)
parser.add_argument("--betm", help="the bet maximum", type=int)
parser.add_argument("--gameid", help=".", type=int)


args = parser.parse_args()
if args.algo:
	algo = args.algo
else:
	algo = "cfrpsrl"
if algo == "cfrpsrl":
	if args.type:
		Type = args.type
	else:
		Type = "default"
if args.cards:
	cards=args.cards
if args.betm:
	betm = args.betm

if args.gameid:
	gameid = args.gameid
for _gameid in range(gameid, 20):
	#print("results/leduc_"+str(cards)+"_"+str(betm)+"_"+str(_gameid)+"_"+algo+"_"+Type +".npz")
	try:
		a = np.load("results/leduc_"+str(cards)+"_"+str(betm)+"_"+str(_gameid)+"_"+algo+"_"+Type +".npz")
	except:
		gamepath="games/leduc_"+str(cards)+"_"+str(betm)+"_"+str(_gameid) 
		print(gamepath)
		game = Game(path=gamepath+".npz")
		print("game info: cards: ", cards, "betmaximum: ", betm, "numHists: ", game.numHists, "algorithm info: ", algo, "explore strategy: ", Type)
		run(game, solvername=algo, Type=Type, resultpath="leduc_"+str(cards)+"_"+str(betm)+"_"+str(_gameid) )