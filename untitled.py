import scipy.io as sio
import numpy as np
import pprint as pp
from matplotlib import pyplot as plt
from collections import namedtuple

shuffledImageEasy = sio.loadmat('shuffledImageEasy.mat')

RGBtilesShuffled = shuffledImageEasy['RGBtilesShuffled']
RGBrearranged = shuffledImageEasy['RGBrearranged']

shuffledTiles = RGBtilesShuffled.reshape(25, 25, 3, 18, 12).astype(np.int32)

height = 18
width = 12

Tile = namedtuple("Tile", ['x', 'y'])

permutation = {}
for i in range(height):
	for j in range(width):
		permutation[Tile(i,j)] = Tile(i,j)

class Tiles:
	def __init__(self, tiles, permutation, height, width):
		self.tiles = tiles
		self.permutation = permutation
		self.height = height
		self.width = width
	def getPermutedTiles(self, tiles):
		permutedTiles = np.copy(tiles)
		for key, val in permutation.iteritems():
			permutedTiles[:, :, :, key.x, key.y] = np.copy(self.tiles[:, :, :, val.x, val.y])
		return permutedTiles
	def getTileColors(self, tile):
		tileColors = permutation[tile]
		return self.tiles[:, :, :, tileColors.x, tileColors.y];
	def asImage(self):
		permutedTiles = self.getPermutedTiles(self.tiles)
		col = []
		for i in range(self.height):
			row = []
			for j in range(self.width):
				if row != []:
					row = np.hstack((row, permutedTiles[:, :, :, i, j]))
				else: 
					row = permutedTiles[:, :, :, i, j]
			if col != []:
				col = np.vstack((col, row))
			else: 
				col = row
		return col.astype(np.uint8)
	def display(self):
		plt.imshow(self.asImage())
		plt.show()
	def swap(self, tile1, tile2):
		permutation[tile1], permutation[tile2] = permutation[tile2], permutation[tile1]
	def getEnergyVMargin(self, tile1):
		tile2 = Tile(tile1.x, tile1.y + 1)
		tile1Colors = self.getTileColors(tile1)
		tile2Colors = self.getTileColors(tile2)
		factor = np.ones(4)
		# factor[0] = 1 + np.abs(tile1.x - (self.height - 2.0) / 2.0)
		# factor[1] = 1 + np.abs(tile1.y - (self.width - 2.0) / 2.0)
		# factor[2] = 1 + np.abs(tile2.x - (self.height - 2.0) / 2.0)
		# factor[3] = 1 + np.abs(tile2.y - (self.width - 2.0) / 2.0)
		return factor.prod() * np.linalg.norm(tile1Colors[:,24,:] - tile2Colors[:,0,:]) ** 2
	def getEnergyHMargin(self, tile1):
		tile2 = Tile(tile1.x + 1, tile1.y)
		tile1Colors = self.getTileColors(tile1)
		tile2Colors = self.getTileColors(tile2)
		factor = np.ones(4)
		# factor[0] = 1 + np.abs(tile1.x - (self.height - 2.0) / 2.0)
		# factor[1] = 1 + np.abs(tile1.y - (self.width - 2.0) / 2.0)
		# factor[2] = 1 + np.abs(tile2.x - (self.height - 2.0) / 2.0)
		# factor[3] = 1 + np.abs(tile2.y - (self.width - 2.0) / 2.0)
		return factor.prod() * np.linalg.norm(tile1Colors[24,:,:] - tile2Colors[0,:,:]) ** 2
	def energy(self):
		energy = 0
		for i in range(self.height):
			for j in range(self.width - 1):
				energy += self.getEnergyVMargin(Tile(i,j))
		for j in range(self.width):
			for i in range(self.height - 1):
				energy += self.getEnergyHMargin(Tile(i,j)) 
		return energy
	
	def getEnergyAround(self, tile):
		energy = np.zeros(4)
		energy[0] = self.getEnergyVMargin(Tile(tile.x, tile.y - 1))
		energy[1] = self.getEnergyVMargin(tile)
		energy[2] = self.getEnergyHMargin(Tile(tile.x - 1, tile.y))
		energy[3] = self.getEnergyHMargin(tile)
		return sum(energy)

	def deltaEnergy(self, tile1, tile2):
		if (tile1 == tile2):
			return 0
		if (tile1.x > tile2.x or tile1.y > tile2.y):
			tile1, tile2 = tile2, tile1			
		inCol =  (tile1.y == tile2.y and tile1.x + 1 == tile2.x)
		inRow =  (tile1.x == tile2.x and tile1.y + 1 == tile2.y)
		
		energyBefore = np.zeros(2)

		energyBefore[0] = self.getEnergyAround(tile1)
		energyBefore[1] = self.getEnergyAround(tile2)

		self.swap(tile1, tile2)	# Swap for computation ease
		energyAfter = np.zeros(2)
		energyAfter[0] = self.getEnergyAround(tile1)
		energyAfter[1] = self.getEnergyAround(tile2)
		self.swap(tile1, tile2)	# Swap back
		
		sumEnergyBefore = energyBefore.sum()
		sumEnergyAfter = energyAfter.sum()

		if inRow:
			sumEnergyBefore -= self.getEnergyVMargin(tile1)
			sumEnergyAfter -= self.getEnergyVMargin(tile2) 
		if inCol:
			sumEnergyBefore -= self.getEnergyHMargin(tile1)
			sumEnergyAfter -= self.getEnergyHMargin(tile2)
		
		return sumEnergyAfter - sumEnergyBefore

	def permutationProposal(self):
		x1 = np.random.randint(1, self.height - 1)
		y1 = np.random.randint(1, self.width - 1)
		x2 = np.random.randint(1, self.height - 1)
		y2 = np.random.randint(1, self.width - 1)
		return [Tile(x1, y1), Tile(x2, y2)]
	def energyAlt(self):
		energy = 0;
		for i in range(1,self.height - 1):
			for j in range(1,self.width - 1):
				energy += self.getEnergyAround(Tile(i,j))

		for j in range(0,self.width - 1):
			energy += 2 * self.getEnergyVMargin(Tile(0,j))

		for j in range(0,self.width - 1):
			energy += 2 * self.getEnergyVMargin(Tile(self.height-1,j))

		for i in range(0,self.height-1):
			energy += 2 * self.getEnergyHMargin(Tile(i,0))

		for i in range(0,self.height - 1):
			energy += 2 * self.getEnergyHMargin(Tile(i,self.width - 1))

		for j in range(1,self.width - 1):
			energy += self.getEnergyHMargin(Tile(0,j))

		for j in range(1,self.width - 1):
			energy += self.getEnergyHMargin(Tile(self.height-2,j))

		for i in range(1,self.height-1):
			energy += self.getEnergyVMargin(Tile(i,0))

		for i in range(1,self.height-1):
			energy += self.getEnergyVMargin(Tile(i,self.width - 2))

		return energy / 2.0
class Annealing:
	def __init__(self, tiles):
		self.tiles = tiles
		self.energy = tiles.energy()
		self.bestEnergy = tiles.energy()
		self.bestPermutation = tiles.permutation
		self.bestT = 0
	def display(self):
		self.tiles.display()
	def step(self, temperature):
		[t1, t2] = tiles.permutationProposal()
		delta = tiles.deltaEnergy(t1, t2)
		if (delta <= 0):
			self.tiles.swap(t1, t2)
			self.energy += delta
		else:
			pAccept = np.exp(- delta / temperature)
			if (np.random.rand() <= pAccept):
				self.tiles.swap(t1, t2)
				self.energy += delta

	def annealing(self, startExp = 6, endExp = -2, numSteps = 10**6):
		temperature = np.logspace(startExp, endExp, numSteps)
		self.bestT = temperature[0]
		for indexTuple, tempVal in np.ndenumerate(temperature):
			index = indexTuple[0]
			self.step(tempVal)
			if (self.energy < self.bestEnergy):
				self.bestPermutation = self.tiles.permutation
				print 'pAccept approx = ' + str(100 * np.exp((self.energy - self.bestEnergy)/tempVal)), ('(1 - newTemp)/ oldTemp = ' + str(1 - tempVal/self.bestT))			
				self.bestT = tempVal
				self.bestEnergy = self.energy
				print('Energy = ' + str(self.bestEnergy) + ' at t = ' + str(tempVal) + ' %done = ' + str(100*index/float(temperature.size)) + '\n')
			if index % 10**5 == 0:
				print('Index = ' + str(index) + ' # sanity check (should be zero) = ' + str(self.energy - self.tiles.energy()) + ' Energy = ' + str(self.energy) + ' Temperature ' + str(tempVal))

## Setup

tiles = Tiles(shuffledTiles, permutation, height, width)
anneal = Annealing(tiles)

#anneal.tiles.display()
print('Starting energy = ' + str(anneal.bestEnergy))