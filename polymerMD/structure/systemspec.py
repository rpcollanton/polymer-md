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
        self.species = species # uses setter! wow, fancy
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
    def label(self):
        return self._species.label
    
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
    
    @property
    def componentlabels(self):
        return self._componentlabels
    
    def addComponent(self, species, N):
        self.components.append(Component(species, N))
        self._componentlabels.append(species.label)
        return self.components[-1]
    
    # This should not be used because components can have identical labels!
    # def componentByLabel(self,label):
    #     return self.components[self._componentlabels.index(label)]
    
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
        count -= self.components[idx_component].numparticles # undo the last step because it overshot N
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
        # len(types) = number of particles

        types = []
        for component in self.components:
            for i in range(component.N):
                for block in component.species.blocks:
                    for j in range(block.length):
                        types.append(block.monomer.label)

        return types
    
    def particleSpeciesTypes(self):
        # returns a list of the type of species a particle is a part of, for all particles
        # len(types) = number of particles

        types = []
        for component in self.components:
            for i in range(component.N):
                for block in component.species.blocks:
                    for j in range(block.length):
                        types.append(component.label)
        
        return types
    
    def speciesTypes(self):
        # returns a list of all species types (defined by their labels)
        # len(types) = number of molecules
        types = []
        for component in self.components:
            for i in range(component.N):
                types.append(component.label)
        
        return types
    
    def indicesByMolecule(self):
        # returns a list of lists where each list corresponds with a given species
        # contains the indices of all particles in that polymer
        
        indices = []
        idx_current = 0
        for component in self.components:
            specieslength = component.species.length
            for i in range(component.N):
                molindices = list(range(idx_current,idx_current+specieslength))
                indices.append(molindices)
                idx_current += specieslength

        return indices

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
    
    def junctions(self):
        junctions = []
        # junctiontypes = [] could add if needed in future for 3 monomer systems
        bonds,bondtypes = self.bonds()
        for bond,bondtype in zip(bonds,bondtypes):
            monomertypes = bondtype.split("-")
            if monomertypes[0] != monomertypes[1]:
                junctions.append(bond)
        return junctions


    
# Workflow:
# Make a system
# Set the box size
# Set the number of components
# For each component, set the number of that component and its molecule spec