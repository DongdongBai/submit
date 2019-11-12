import numpy as np
from LeducHoldem import Game
import copy
import queue
import utils
from utils import RegretSolver, exploitability, generateOutcome, RegretSolverPlus, simulate
import time


class FICPLAY:
	def __init__(self, game):
		self.game = game

		self.trajs = [[], []]

		self.isetflag = [-1 * np.ones(game.numIsets[0]), -1 * np.ones(game.numIsets[1])]

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

		self.Q = [[], []]
		for owner in range(2):
			for iset in range(game.numIsets[owner]):
				self.Q[owner].append(np.zeros(game.nactsOnIset[owner][iset]))

	def updateAll(self):
		game = self.game

		self.round += 1
		

		learningrate = 0.05 / (1.0 + 0.003 * np.sqrt(1.0 * self.round))
		temperature = (1.0 + 0.02 * np.sqrt(1.0 * self.round))


		def simulate(h, histtraj):#simulate strategies and add data to replay memory
			player = game.playerOfHist[h]
			curstgy = None
			if player == 2:
				curstgy = game.chanceprob[h]
			else:
				iset = game.Hist2Iset[player][h]
				curstgy = self.stgy[player][iset]
			a = np.random.choice(game.nactsOnHist[h], p=curstgy)
			nxth = game.histSucc[h][a]
			if game.isTerminal[nxth]:
				r = game.simulate(nxth)
				histtraj.append((h,a,nxth, r[0]))
			else:
				histtraj.append((h, a, nxth, 0.0))
				simulate(nxth, histtraj)

		def translate_traj(owner, histtraj, isettraj):
			ids = []
			for i in range(len(histtraj)):
				h = histtraj[i][0]
				if game.playerOfHist[h] == owner:
					ids.append(i)
			if len(ids) == 0:
				return []
			for i, inds in enumerate(ids):
				h = histtraj[inds][0]
				a = histtraj[inds][1]
				iset = game.Hist2Iset[owner][h]
				if i == len(ids) - 1:
					niset = -1
					rews = 0.0
					for j in range(inds, len(histtraj)):
						rews += histtraj[j][3]
					if owner == 1:
						rews *= -1
					isettraj.append((iset, a, -1, rews))
				else:
					nh = histtraj[ids[i + 1]][0]
					niset = game.Hist2Iset[owner][nh]
					isettraj.append((iset, a, niset, 0.0))

		for _fsdf in range(2):
			histtraj = []
			simulate(0, histtraj)
			isettrajs = [[], []]
			translate_traj(0, histtraj, isettrajs[0])
			translate_traj(1, histtraj, isettrajs[1])

			self.trajs[0].append(isettrajs[0])
			self.trajs[1].append(isettrajs[1])

		def updQtraj(owner, isettraj, lr):
			for iset, a, niset, rew in isettraj:
				if niset == -1:
					self.Q[owner][iset][a] = (1.0 - lr) * self.Q[owner][iset][a] + lr * rew
				else:
					self.Q[owner][iset][a] = (1.0 - lr) * self.Q[owner][iset][a] + lr * self.Q[owner][iset].max()

		for k in range(30):
			for p in range(2):
				trajid = np.random.randint(0, len(self.trajs[p]))
				updQtraj(p, self.trajs[p][trajid], learningrate)

		self.genStgy(temperature)
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


	def genStgy(self, temp):
		game = self.game
		for owner in range(2):
			for iset in range(game.numIsets[owner]):
				if game.playerOfIset[owner][iset] == owner:
					nacts = game.nactsOnIset[owner][iset]
					newstgy = np.zeros(game.nactsOnIset[owner][iset]) 
					for a in range(nacts):
						newstgy[a] = np.exp(self.Q[owner][iset][a] * temp)
					newstgy /= newstgy.sum()
					self.stgy[owner][iset] = newstgy

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
game = Game(bidmaximum=4)
print("game", game.numHists, game.numIsets)
cfr =FICPLAY(game)
for i in range(100000):
	cfr.updateAll()
	print(i, cfr.getExploitability())
"""