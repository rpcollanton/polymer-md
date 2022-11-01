import types
import numpy as np

class MonomerSpec:

    def __init__(self, label, l, uniqueid):
        self.l = l
        self.label = label
        self._uniqueid = uniqueid

    @property
    def uniqueid(self):
        return self._uniqueid

class BlockSpec:

    def __init__(self):
        return
    
    @property
    def monomer(self):
        return self._monomer

    @monomer.setter
    def monomer(self,mnr):
        self._monomer = mnr
        return
    
    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        return

class LinearPolymerSpec:

    def __init__(self, monomers, lengths):
        
        self.nBlocks = len(monomers)
        for i in range(self.nBlocks):
            self._blocks[i].monomer = monomers[i]
            self._blocks[i].length = lengths[i]
    
    @property
    def nBlocks(self):
        return self._nBlocks
    
    @nBlocks.setter
    def nBlocks(self, value):
        self._nBlocks = value
        self._blocks = [BlockSpec() for i in range(value)]
        return    
    
    @property
    def blocks(self):
        return self._blocks
    
    @property
    def length(self):
        return np.sum([block.length for block in self.blocks])

    @property
    def label(self):
        return ''.join([block.monomer.label for block in self.blocks])

    @property
    def bonds(self):
        # this is specific to a linear polymer
        # all the bonds, assuming that the first particle is indexed at 0 
        bonds = []
        Ntot = 0
        for idxblock,block in enumerate(self.blocks):
            # within the block
            for i in range(1,block.length):
                bonds.append([Ntot + (i-1), Ntot + i])
            # connect the blocks
            if idxblock < (self.nBlocks-1):
                bonds.append([Ntot + block.length - 1, Ntot + block.length])
            # chain length so far
            Ntot += block.length
        
        return bonds

    @property
    def bondtypes(self):
        # this is specific to a linear polymer
        # all the bond types
        bondtypes = []
        for idxblock, block in enumerate(self.blocks):
            # within the block
            for i in range(1,block.length):
                bondtypes.append('{:s}-{:s}'.format(block.monomer.label, block.monomer.label))
            # connect the blocks
            if idxblock < (self.nBlocks-1):
                uniqueids = [block.monomer.uniqueid, self.blocks[idxblock+1].monomer.uniqueid]
                labels = [block.monomer.label, self.blocks[idxblock+1].monomer.label]
                minID = np.argmin(uniqueids)
                maxID = np.argmax(uniqueids)
                bondtypes.append('{:s}-{:s}'.format(labels[minID], labels[maxID]))

        return bondtypes

class Component:

    def __init__(self, species, N):
        self.species = species
        self.N = N
        return

    @property 
    def species(self):
        return self._species
    
    @species.setter
    def species(self, molspec):
        self._species = molspec
        return
    
    @property 
    def N(self):
        return self._N
    
    @N.setter
    def N(self, value):
        self._N = value
        return
    
    @property
    def numparticles(self):
        return self.N*self.species.length
    
class Box:

    def __init__(self, lengths, tilts=[0,0,0]):

        self.lengths = lengths
        self.tilts = tilts

        return

class System:

    def __init__(self):
        self._components = []
        self._componentlabels = []
        self._monomers = []
        self._monomerlabels = []
        return
    
    @property
    def box(self):
        return self._box.lengths + self._box.tilts
    
    @box.setter
    def box(self, size):
        if len(size) == 6:
            self._box = Box(size[0:3], size[3:])
        else:
            self._box = Box(size)
        return
    
    @property
    def nComponents(self):
        return len(self._components)

    @property
    def components(self):
        return self._components
    
    def addComponent(self, species, N):
        self.components.append(Component(species, N))
        self._componentlabels.append(species.label)
        return self.components[-1]
    
    def componentByLabel(self,label):
        return self.components[self._componentlabels.index(label)]
    
    @property
    def nMonomers(self):
        return len(self._monomers)

    @property
    def monomers(self):
        return self._monomers
    
    @property
    def monomerlabels(self):
        return self._monomerlabels
    
    def addMonomer(self, label, l):
        uniqueid = self.nMonomers
        self.monomers.append(MonomerSpec(label, l, uniqueid))
        self._monomerlabels.append(label)
        return self.monomers[-1]

    def monomerByLabel(self,label):
        return self.monomers[self._monomerlabels.index(label)]
    
    @property
    def numparticles(self):
        N = 0
        for component in self.components:
            N += component.numparticles
        return N
    
    def particleType(self, N):

        # takes a particle index
        # finds the right component
        count = 0
        idx_component = -1
        while count < N:
            idx_component += 1
            count += self.components[idx_component].numparticles
        count -= self.components[idx_component].numparticles 
        # Note: count will finish being equal to the number of particles preceding this component

        # find the idx on the species chain
        idx_species = (N - count) % self.component[idx_component].species.length

        # find the right block
        count = 0
        idx_block = -1
        while count < idx_species:
            idx_block += 1
            count += self.components[idx_component].species.blocks[idx_block].length

        # returns the monomer associated with this block
        return self.components[idx_component].species.blocks[idx_block].monomer
    
    def particleTypes(self):
        # returns a list of all particle types in order
        # faster than running particleType for each particle

        types = []
        for component in self.components:
            for i in range(component.N):
                for block in component.species.blocks:
                    for j in range(block.length):
                        types.append(block.monomer.label)

        return types
    
    def bonds(self):

        bonds = []
        bondtypes = []
        idx_start = 0
        for component in self.components:
            for i in range(component.N):
                bonds += ( np.array(component.species.bonds) + idx_start ).tolist()
                bondtypes += component.species.bondtypes
                idx_start += component.species.length
        
        return bonds, bondtypes

    
# Workflow:
# Make a system
# Set the box size
# Set the number of components
# For each component, set the number of that component and its molecule spec