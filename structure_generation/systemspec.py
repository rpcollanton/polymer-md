import numpy as np

class MonomerSpec:

    def __init__(self, label, l):
        self.l = l
        self.label = label

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

    def __init__(self, nBlocks, types, lengths, bondlengths):
        
        self.nBlocks = nBlocks
        for i in range(self.nBlocks):
            mnr = MonomerSpec(types[i], bondlengths[i])
            self._blocks[i].monomer = mnr
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

class Component:

    def __init__(self):
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

    def __init__(self, lengths, angles=[np.pi/2, np.pi/2, np.pi/2]):

        self.lengths = lengths
        self.angles = angles

        return

class System:

    def __init__(self):
        return
    
    @property
    def box(self):
        return self._box.lengths + self._box.angles
    
    @box.setter
    def box(self, size):
        self._box = Box(size[0:3], size[3:])
        return
    
    @property
    def nComponents(self):
        return len(self._components)

    @nComponents.setter
    def nComponents(self, value):
        self._components = [Component() for i in range(value)]
        return
    
    @property
    def components(self):
        return self._components
    
    @property
    def numparticles(self):
        N = 0
        for component in self.components:
            N += component.numparticles
        return N
    


# Workflow:
# Make a system
# Set the box size
# Set the number of components
# For each component, set the number of that component and its molecule spec