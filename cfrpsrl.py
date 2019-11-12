import numpy as np
from LeducHoldem import Game
import copy
import queue
import utils
from utils import RegretSolver, exploitability, generateOutcome, RegretSolverPlus, simulate
import time


class CFRPSRL:
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

		transpsrl, rewpsrl = game.resample()
		transvalidation, rewvalidation = game.resample()

		def avgchance(h, curtrans, currew, w, sumtrans, sumrew, sumw):
			sumw[h] += w 
			term = game.isTerminal[h]
			player = game.playerOfHist[h]
			if term == True:

				sumrew[h] += (w * currew[h][0], w * currew[h][1])
				return
			if player == 2:
				for a in range(game.nactsOnHist[h]):
					avgchance(game.histSucc[h][a], curtrans, currew, w * curtrans[h][a], sumtrans, sumrew, sumw)
				sumtrans[h] += w * curtrans[h]
			else:
				for a in range(game.nactsOnHist[h]):
					avgchance(game.histSucc[h][a], curtrans, currew, w , sumtrans, sumrew, sumw)

		avgchance(0, transpsrl, rewpsrl, 1.0, self.sampledtrans1, self.sampledrews1, self.weight1)


		self.update(0, 0, [np.ones(1), np.ones(1)], [0], rewpsrl, transpsrl)#the CFR algorithm
		self.update(1, 0, [np.ones(1), np.ones(1)], [0], rewpsrl, transpsrl)#the CFR algorithm


		def updStgy(owner, iset):
			if self.isetflag[owner][iset] != self.round:
				return
			player = game.playerOfIset[owner][iset]
			if player == owner:
				self.stgy[owner][iset] = self.solvers[owner][iset].curstgy.copy()
			for nxtiset in game.isetSucc[owner][iset]:
				updStgy(owner, nxtiset)
		updStgy(0, 0)
		updStgy(1, 0)
		

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

		avgstgy = self.avgstgyprofile()

		def getExploreStgy(owner, iset, explore_stgy, oppstgy, ds_c1, ds_c2):
			rew1, trans1, reachp1= ds_c1
			rew2, trans2, reachp2 = ds_c2
			hists = game.Iset2Hists[owner][iset]
			if game.isTerminal[hists[0]] == True:
				return
			player = game.playerOfIset[owner][iset]
			if player == owner:

				nacts = game.nactsOnIset[owner][iset]

				outcome1 = np.zeros(nacts)
				outcome2 = np.zeros(nacts)
				for a in range(nacts):
					getExploreStgy(owner, game.isetSucc[owner][iset][a], explore_stgy, oppstgy, ds_c1, ds_c2)

					for h in hists:
						outcome1[a] += reachp1[h] * rew1[game.histSucc[h][a]][owner]
						outcome2[a] += reachp2[h] * rew2[game.histSucc[h][a]][owner]
				a_star = np.argmax(outcome1 - outcome2)
				_stgy = np.zeros(nacts)
				_stgy[a_star] = 1

				explore_stgy[iset] = _stgy

				for h in hists:
					rew1[h] = rew1[game.histSucc[h][a_star]]
					rew2[h] = rew2[game.histSucc[h][a_star]]

			else:
				truenacts = game.nactsOnHist[hists[0]]
				obsnacts = game.nactsOnIset[owner][iset]
				for h in hists:
					_stgy1 = None
					_stgy2 = None
					if player == 2:
						_stgy1 = trans1[h]
						_stgy2 = trans2[h]
					else:
						piset = game.Hist2Iset[player][h]
						_stgy1 = oppstgy[piset]
						_stgy2 = oppstgy[piset]
					nactsh = game.nactsOnHist[h]
					for a in range(nactsh):
						reachp1[game.histSucc[h][a]] = reachp1[h] * _stgy1[a]
						reachp2[game.histSucc[h][a]] = reachp2[h] * _stgy2[a]


				for a in range(obsnacts):
					getExploreStgy(owner, game.isetSucc[owner][iset][a], explore_stgy, oppstgy, ds_c1,ds_c2)
				for h in hists:
					_stgy1 = None
					_stgy2 = None
					if player == 2:
						_stgy1 = trans1[h]
						_stgy2 = trans2[h]
					else:
						piset = game.Hist2Iset[player][h]
						_stgy1 = oppstgy[piset]
						_stgy2 = oppstgy[piset]
					nactsh = game.nactsOnHist[h]
					for a in range(nactsh):   
						rew1[h] += rew1[game.histSucc[h][a]] * _stgy1[a]
						rew2[h] += rew2[game.histSucc[h][a]] * _stgy2[a]
			if iset == 0:
				pass
				#print("check", rew1[0] - rew2[0], rew1[0], rew2[0])
								

		#avgchrew = np.zeros(game.numHists).tolist()
		#def getavgchance(h, avgchrew, avgchtrans):


		#getavgchance(0, avgchrew, avgchtrans)
		prob1 = np.ones(game.numHists)
		prob2 = np.ones(game.numHists)
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

		avgchtrans = copy.deepcopy(self.sampledtrans1)
		avgchrew = copy.deepcopy(self.sampledrews1)

		for h in range(game.numHists):
			if game.isTerminal[h]:
				avgchrew[h] /= self.weight1[h]
			if game.playerOfHist[h] == 2:
				avgchtrans[h] /=self.weight1[h]

		if self.Type == "default":
			getExploreStgy(0, 0, explore_stgy[0], avgstgy[1], (rewvalidation, transvalidation, prob2), (avgchrew, avgchtrans, prob1))
			getExploreStgy(1, 0, explore_stgy[1], avgstgy[0], (rewvalidation, transvalidation, prob2), (avgchrew, avgchtrans, prob1))
			for t in range(1):
				simulate(game, 0, [explore_stgy[0], self.stgy[1]])
				simulate(game, 0, [self.stgy[0], explore_stgy[1]])

		if self.Type == "br_dirc":
			getExploreStgy(0, 0, explore_stgy[0], avgstgy[1], (rewvalidation, transvalidation, prob2), (avgchrew * 0.0, avgchtrans, prob1))
			getExploreStgy(1, 0, explore_stgy[1], avgstgy[0], (rewvalidation, transvalidation, prob2), (avgchrew * 0.0, avgchtrans, prob1))
			for t in range(1):
				simulate(game, 0, [explore_stgy[0], self.stgy[1]])
				simulate(game, 0, [self.stgy[0], explore_stgy[1]])


		if self.Type == "ordinary":
			simulate(game, 0, self.stgy)
			simulate(game, 0, self.stgy)
		if self.Type == "random":
			for i in range(2):
				for iset in range(game.numIsets[i]):
					pl = game.playerOfIset[i][iset]
					explore_stgy[i].append(0)
					if pl == i:
						nacts = game.nactsOnIset[i][iset]
						explore_stgy[i][iset] = np.ones(nacts)/nacts
			simulate(game, 0, explore_stgy)
			simulate(game, 0, explore_stgy)



	def update(self, owner, iset, probs, histories, rewards, chanceprob):
		self.isetflag[owner][iset] = self.round
		if len(histories) == 0:
			return np.zeros(0)
		
		self.nodestouched += len(histories)
		game = self.game
		player = game.playerOfIset[owner][iset] 

		if game.isTerminal[game.Iset2Hists[owner][iset][0]]:
			ret = np.zeros(len(histories))
			for hid, h in enumerate(histories):
				ret[hid] = rewards[h][owner]
			return  ret

		if player == owner:
			nacts = game.nactsOnIset[owner][iset]
			ret = np.zeros(len(histories))
			cfv = np.zeros(game.nactsOnIset[owner][iset])
			for a in range(nacts):
				nxthistories = []
				nxtprobs = [probs[0].copy(), probs[1].copy()]
				nxtprobs[owner] *= self.stgy[owner][iset][a]
				for h in histories:
					nxthistories.append(game.histSucc[h][a])
				tmpr = self.update(owner, game.isetSucc[owner][iset][a], nxtprobs, nxthistories, rewards, chanceprob)
				for nhid, nh in enumerate(nxthistories):
					cfv[a] += probs[1 - owner][nhid] * tmpr[nhid]
					ret[nhid] += tmpr[nhid] * self.stgy[player][iset][a]

			self.solvers[owner][iset].receive(cfv, weight=probs[owner][0])
			return ret

		else:
			obsnact = game.nactsOnIset[owner][iset]
			nacts = np.array(list(map(lambda _u: game.nactsOnHist[_u], histories)))
			truenact = int(nacts.max())

			if truenact == obsnact:
				ret = np.zeros(len(histories))
				for i in range(truenact):
					nxtiset = game.isetSucc[owner][iset][i]
					nxtprobs = [[],[]]
					nxthistories = []
					_ps = []
					_ids = []
					for j, h in enumerate(histories):
						player = game.playerOfHist[h]
						stgy = None
						if player == 2:
							stgy = chanceprob[h]
						else:
							piset = game.Hist2Iset[player][h]
							stgy = self.stgy[player][piset]
						if probs[1 - owner][j] * stgy[i] < 1e-5:
							pass
						else:
							nxtprobs[1 - owner].append(probs[1 - owner][j] * stgy[i])
							nxthistories.append(game.histSucc[h][i])
							nxtprobs[owner].append(probs[owner][j])
							_ps.append(stgy[i])
							_ids.append(j)
					nxtprobs[0] = np.array(nxtprobs[0])
					nxtprobs[1] = np.array(nxtprobs[1])
					tmpr = self.update(owner, nxtiset, nxtprobs, nxthistories, rewards, chanceprob)
					if len(_ids) > 0:
						for _id, _r in enumerate(tmpr):
							ret[_ids[_id]] += _ps[_id] * _r
				return ret

			else:
				ret = np.zeros(len(histories))
				nxtprobs = [[], []]
				nxthistories = []
				_ps = []
				_ids = []
				for aid in range(truenact):
					for hid, h in enumerate(histories):


						player = game.playerOfHist[h]
						stgy = None
						if player == 2:
							stgy = chanceprob[h]
						else:
							piset = game.Hist2Iset[player][h]
							stgy = self.stgy[player][piset]
						if probs[1 - owner][hid] * stgy[aid] < 1e-5:#player == 2 and 
							pass
						else:
							nxtprobs[1 - owner].append(probs[1 - owner][hid] * stgy[aid])
							nxthistories.append(game.histSucc[h][aid])
							nxtprobs[owner].append(probs[owner][hid])
							_ps.append(stgy[aid])
							_ids.append(hid)
				nxtiset = game.isetSucc[owner][iset][0]
				nxtprobs[0] = np.array(nxtprobs[0])
				nxtprobs[1] = np.array(nxtprobs[1])
				tmpr = self.update(owner, game.isetSucc[owner][iset][0], nxtprobs, nxthistories, rewards, chanceprob)
				for _i, _r in enumerate(tmpr):
					ret[_ids[_i]] += _r * _ps[_i]
				return ret

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

"""