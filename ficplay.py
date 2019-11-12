import numpy as np
from LeducHoldem import Game
import copy
import queue
import utils
from utils import RegretSolver, exploitability, generateOutcome, RegretSolverPlus, simulate
import time



class FICPLAY:
	def __init__(self, game, Type="regretmatching"):
		self.game = game
		self.Type = Type
		Solver = None
		if Type == "regretmatching":
			Solver=RegretSolver
		else:
			Solver=RegretSolverPlus


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

		transpsrl, rewpsrl = game.resample()




		avgstgy = self.avgstgyprofile()

		def getExploreStgy(owner, iset, explore_stgy, oppstgy, ds_c):
			rew, trans, reachp= ds_c
			hists = game.Iset2Hists[owner][iset]
			if game.isTerminal[hists[0]] == True:
				return
			player = game.playerOfIset[owner][iset]
			if player == owner:
				nacts = game.nactsOnIset[owner][iset]
				outcome = np.zeros(nacts)
				for a in range(nacts):
					getExploreStgy(owner, game.isetSucc[owner][iset][a], explore_stgy, oppstgy, ds_c)

					for h in hists:
						outcome[a] += reachp[h] * rew[game.histSucc[h][a]][owner]
				a_star = np.argmax(outcome)
				_stgy = np.zeros(nacts)
				_stgy[a_star] = 1

				explore_stgy[iset] = _stgy

				for h in hists:
					rew[h] = rew[game.histSucc[h][a_star]]

			else:
				truenacts = game.nactsOnHist[hists[0]]
				obsnacts = game.nactsOnIset[owner][iset]
				for h in hists:
					_stgy = None
					if player == 2:
						_stgy = trans[h]
					else:
						piset = game.Hist2Iset[player][h]
						_stgy = oppstgy[piset]
					nactsh = game.nactsOnHist[h]
					for a in range(nactsh):
						reachp[game.histSucc[h][a]] = reachp[h] * _stgy[a]


				for a in range(obsnacts):
					getExploreStgy(owner, game.isetSucc[owner][iset][a], explore_stgy, oppstgy, ds_c)
				for h in hists:
					_stgy = None
					if player == 2:
						_stgy = trans[h]
					else:
						piset = game.Hist2Iset[player][h]
						_stgy = oppstgy[piset]
					nactsh = game.nactsOnHist[h]
					for a in range(nactsh):   
						rew[h] += rew[game.histSucc[h][a]] * _stgy[a]
	

		prob = np.ones(game.numHists)
		explore_stgy = [[],[]]

		for i, iset in enumerate(range(game.numIsets[0])):
			nact = game.nactsOnIset[0][iset]
			if game.playerOfIset[0][iset] == 0:
				explore_stgy[0].append(np.ones(nact) / nact)
			else:
				explore_stgy[0].append(np.ones(0))
		for i, iset in enumerate(range(game.numIsets[1])):
			nact = game.nactsOnIset[1][iset]
			if game.playerOfIset[1][iset] == 1:
				explore_stgy[1].append(np.ones(nact) / nact)
			else:
				explore_stgy[1].append(np.ones(0))

		getExploreStgy(0, 0, explore_stgy[0], avgstgy[1], (rewpsrl, transpsrl, prob))
		getExploreStgy(1, 0, explore_stgy[1], avgstgy[0], (rewpsrl, transpsrl, prob))



		simulate(game, 0, explore_stgy)
		simulate(game, 0, explore_stgy)


		def updStgy(owner, iset, expstgy):
			player = game.playerOfIset[owner][iset]
			if player == owner:
				self.stgy[owner][iset] = expstgy[owner][iset].copy()
			for nxtiset in game.isetSucc[owner][iset]:
				updStgy(owner, nxtiset, expstgy)
		updStgy(0, 0, explore_stgy)
		updStgy(1, 0, explore_stgy)
		
		def updSumstgy(owner, iset, prob = 1.0):
			player = game.playerOfIset[owner][iset]
			if player == owner:
				self.sumstgy[owner][iset] += prob * self.stgy[player][iset]
				for aid, nxtiset in enumerate(game.isetSucc[owner][iset]):
					if prob * self.stgy[player][iset][aid] > 1e-8:
						updSumstgy(owner, nxtiset, prob * self.stgy[player][iset][aid])
			else:
				for aid, nxtiset in enumerate(game.isetSucc[owner][iset]):
					updSumstgy(owner, nxtiset, prob)
					
		updSumstgy(0, 0)
		updSumstgy(1, 0)


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
game = Game()
print("game", game.numHists, game.numIsets)
cfr =CFRPSRL(game)
for i in range(100000):
	cfr.updateAll()
	print(i, cfr.getExploitability())
"""