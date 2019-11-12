import numpy as np
from LeducHoldem import Game
import copy
import queue
import utils
from utils import RegretSolver, exploitability, generateOutcome, RegretSolverPlus, simulate
import time


class MCCFR_OS:
	def __init__(self, game, Type="default"):
		self.game = game
		self.Type = Type
		Solver = None
		Solver=RegretSolver


		self.isetflag = [-1 * np.ones(game.numIsets[0]), -1 * np.ones(game.numIsets[1])]


		self.solvers = []
		self.solvers.append(list(map(lambda x:  Solver(game.nactsOnIset[0][x]), range(game.numIsets[0]))))
		self.solvers.append(list(map(lambda x:  Solver(game.nactsOnIset[1][x]), range(game.numIsets[1]))))
		self.stgy = [[], []]
		for i, iset in enumerate(range(game.numIsets[0])):
			nact = game.nactsOnIset[0][iset]
			if game.playerOfIset[0][iset] == 0:
				self.stgy[0].append(np.ones(nact) / nact)
			else:
				self.stgy[0].append(np.ones(0))

		for i, iset in enumerate(range(game.numIsets[1])):
			nact = game.nactsOnIset[1][iset]
			if game.playerOfIset[1][iset] == 1:
				self.stgy[1].append(np.ones(nact) / nact)
			else:
				self.stgy[1].append(np.ones(0))

		self.outcome,self.reward = generateOutcome(game, self.stgy)
		self.nodestouched = 0
		self.round = -1


		self.sumstgy = [[], []]
		for i, iset in enumerate(range(game.numIsets[0])):
			nact = game.nactsOnIset[0][iset]
			if game.playerOfIset[0][iset] == 0:
				self.sumstgy[0].append(np.ones(nact) / nact)
			else:
				self.sumstgy[0].append(np.ones(0))
		for i, iset in enumerate(range(game.numIsets[1])):
			nact = game.nactsOnIset[1][iset]
			if game.playerOfIset[1][iset] == 1:
				self.sumstgy[1].append(np.ones(nact) / nact)
			else:
				self.sumstgy[1].append(np.ones(0))

		self.sampledtrans1 = list(map(lambda _u: np.zeros(game.nactsOnHist[_u]), range(game.numHists)))
		self.sampledrews1 = list(map(lambda _u: np.zeros(2), range(game.numHists)))
		self.weight1 = np.zeros(game.numHists)

	def updateAll(self):
		game = self.game

		self.round += 1


		def updSumstgy(owner, iset, prob = 1.0):
			player = game.playerOfIset[owner][iset]
			if player == owner:
				self.sumstgy[owner][iset] += prob * self.solvers[player][iset].take()#.curstgy
				for aid, nxtiset in enumerate(game.isetSucc[owner][iset]):
					if prob * self.stgy[player][iset][aid] > 1e-8:
						updSumstgy(owner, nxtiset, prob * self.solvers[player][iset].take()[aid])#self.stgy[player][iset][aid]
			else:
				for aid, nxtiset in enumerate(game.isetSucc[owner][iset]):
					updSumstgy(owner, nxtiset, prob)
		updSumstgy(0, 0)
		updSumstgy(1, 0)

		avgstgy = self.avgstgyprofile()



		#simulate(game, 0, self.stgy)
		#simulate(game, 0, self.stgy)

		def outcome_simulate(game, owner, h, eps=0.1):#, stgy_prof
			if game.isTerminal[h]:
				return game.simulate(h)
				#return
			player = game.playerOfHist[h]
			if player == 2:
				nh = game.simulate(h)
				return outcome_simulate(game, owner, nh)

			piset = game.Hist2Iset[player][h]
			stgy = self.solvers[player][piset].take()#curstgy #stgy_prof[player][piset]
			#simulate(game, game.histSucc[h][a], stgy_prof)

			if player == owner:
				noise = eps * np.ones(stgy.shape[0])
				_stgy = stgy + noise
				_stgy /= _stgy.sum()
				a = np.random.choice(game.nactsOnHist[h], p=_stgy) 
				dim = self.solvers[owner][piset].dim
				cfv = np.zeros(dim)
				ret = outcome_simulate(game, owner, game.histSucc[h][a])
				#print(ret)
				cfv[a] = ret[owner] / _stgy[a]
				self.solvers[owner][piset].receive(cfv)
				return ret / _stgy[a]
			else:
				a = np.random.choice(game.nactsOnHist[h], p=stgy) 
				return outcome_simulate(game, owner, game.histSucc[h][a])
		outcome_simulate(game, 0, 0)
		outcome_simulate(game, 1, 0)


	def avgstgyprofile(self):

		stgy_prof = []
		def avg(_x):
			s = np.sum(_x)
			l = _x.shape[0]
			if s < 1e-5:
				return np.ones(l) / l
			return _x / s
		stgy_prof.append(list(map( lambda _x: avg(_x), self.sumstgy[0] )))
		stgy_prof.append(list(map( lambda _x: avg(_x), self.sumstgy[1] )))
		return stgy_prof

	def getExploitability(self):
		stgy_prof = self.avgstgyprofile()
		return exploitability(self.game, stgy_prof)
"""
bidmaximum = 4
gamepath = "leduc_3_" + str(bidmaximum) + ".npz"
game = Game(path=gamepath)#bidmaximum=bidmaximum)
print("game", game.numHists, game.numIsets)
cfr =CFRPSRL(game)

expls = []
for i in range(100000):
	cfr.updateAll()
	ae = cfr.getExploitability()
	print(i, Type, ae)
	expls.append(ae)
	if i % 50 == 0:
		np.savez("leduc_"+str(bidmaximum) + "cfrpsrl_"+Type, expls=np.array(expls))

game = Game(cards = 2, bidmaximum=1)
print(game.numHists)
mcos = MCCFR_OS(game)
for t in range(100000):
	mcos.updateAll()
	if t % 100 == 0:
		print(mcos.getExploitability())
"""