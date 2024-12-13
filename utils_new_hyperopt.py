import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
import numpy as np
import openbabel as ob
from rdkit.Chem import AllChem
import tensorflow as tf

def obsmitosmile(smi):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = ob.OBMol()
    conv.ReadString(mol, smi)  
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile

def calculate_bond_length(positions, atom1_idx, atom2_idx):
    pos1 = positions[atom1_idx]
    pos2 = positions[atom2_idx]
    bond_length = np.linalg.norm(pos1 - pos2)  
    return bond_length

def calculate_bond_angle(positions, atom1_idx, atom2_idx, atom3_idx):
    pos1 = positions[atom1_idx]
    pos2 = positions[atom2_idx]
    pos3 = positions[atom3_idx]
    
    vec1 = pos1 - pos2
    vec2 = pos3 - pos2

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    dot_product = np.dot(vec1, vec2)

    if norm_vec1 == 0 or norm_vec2 == 0 or np.isnan(dot_product) or np.isinf(dot_product) :
      cos_theta = -0.3578030616898111
      #print(norm_vec1,norm_vec2,dot_product)

    else:
      cos_theta = dot_product / (norm_vec1 * norm_vec2)

    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  
    angle_deg = np.degrees(angle_rad)  
    if angle_deg > 90:
        angle_deg=180-angle_deg

    return angle_deg


def add_bond_nodes(adjacency_matrix, molecule):

    num_atoms = len(adjacency_matrix)
    num_bond_nodes = molecule.GetNumBonds()
    total_nodes = num_atoms + num_bond_nodes

    extended_adj_matrix = np.zeros((total_nodes, total_nodes), dtype=float)

    extended_adj_matrix[:num_atoms, :num_atoms] = adjacency_matrix

    for bond1 in molecule.GetBonds():
        atom1, atom2 = bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx()
        
        extended_adj_matrix[atom1, num_atoms + bond1.GetIdx()] = 1
        extended_adj_matrix[num_atoms + bond1.GetIdx(), atom1] = 1
        extended_adj_matrix[atom2, num_atoms + bond1.GetIdx()] = 1
        extended_adj_matrix[num_atoms + bond1.GetIdx(), atom2] = 1
  
        for bond2 in molecule.GetBonds():
            if bond1.GetIdx() != bond2.GetIdx():  
                if bond1.GetBeginAtomIdx() == bond2.GetBeginAtomIdx() or bond1.GetBeginAtomIdx() == bond2.GetEndAtomIdx() or \
                        bond1.GetEndAtomIdx() == bond2.GetBeginAtomIdx() or bond1.GetEndAtomIdx() == bond2.GetEndAtomIdx():
                    extended_adj_matrix[num_atoms + bond1.GetIdx(), num_atoms + bond2.GetIdx()] = 1
                    extended_adj_matrix[num_atoms + bond2.GetIdx(), num_atoms + bond1.GetIdx()] = 1
    np.fill_diagonal(extended_adj_matrix, 1)

    return extended_adj_matrix

def calculate_all_bond_lengths_and_angles(positions, adj_matrix,mol):
    num_atoms = len(positions)
    bond_lengths = []
    bond_angles = []

    extended_adj_matrix = add_bond_nodes(adj_matrix, mol)

    #print(extended_adj_matrix)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if adj_matrix[i, j] == 1:  
                bond_length = calculate_bond_length(positions, i, j)
                if bond_length != 0:
                  bond_length=1.9/bond_length      
                else:
                  bond_length=1
                bond_length_rounded = round(bond_length, 2)
                #print(bond_length_rounded)
                bond_lengths.append(bond_length_rounded)

                
                extended_adj_matrix[i, j] = bond_length_rounded
                extended_adj_matrix[j, i] = bond_length_rounded  
                distance_matrix=extended_adj_matrix.copy()

                distance_matrix[num_atoms:, num_atoms:] = 0
     
                for k in range(num_atoms):

                    if adj_matrix[i, k] == 1 and k != j :
                        #print(k, i, j)
                        angle=calculate_bond_angle(positions, k, i, j)
                        angle=angle / 90
                        if angle != 0:
                          angle=0.5/angle  
                        else:
                          angle=1
                        angle = np.round(angle, 2)
                        #print(angle)
                        bond_angles.append(angle)

                        for bond1 in mol.GetBonds():
                                if (bond1.GetBeginAtomIdx() == k and bond1.GetEndAtomIdx() == i) or (bond1.GetBeginAtomIdx() == i and bond1.GetEndAtomIdx() == k) :
                                    #print(1)
                                    for bond2 in mol.GetBonds():
                                        if (bond2.GetBeginAtomIdx() == i and bond2.GetEndAtomIdx() == j) or (bond2.GetBeginAtomIdx() == j and bond2.GetEndAtomIdx() == i) :
                                            #print(2)
                                            if bond1.GetIdx() != bond2.GetIdx():
                                                #print(3)
                                                extended_adj_matrix[num_atoms + bond1.GetIdx(), num_atoms + bond2.GetIdx()] = angle
                                                extended_adj_matrix[num_atoms + bond2.GetIdx(), num_atoms + bond1.GetIdx()] = angle
                angle_matrix = extended_adj_matrix.copy()
                angle_matrix[:num_atoms, :num_atoms] = 0


    #return  distance_matrix,angle_matrix
    return  extended_adj_matrix


def smiles2adjoin(smiles,explicit_hydrogens=True,canonical_atom_order=False):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()

    atoms_list = []
    bonds_list = []
    atoms_bonds_list = []

    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())

    num_bonds = mol.GetNumBonds()
    for j in range(num_bonds):
        bond = mol.GetBondWithIdx(j)
        bond_type = bond.GetBondTypeAsDouble()
        if bond_type == 1.0:
            bonds_list.append("single")
        elif bond_type == 2.0:
            bonds_list.append("double")   
        elif bond_type == 3.0:
            bonds_list.append("triple")
        elif bond_type == 1.5:
            bonds_list.append("ben_double")

    atoms_bonds_list=atoms_list+bonds_list

    adj_matrix_ethanol = Chem.GetAdjacencyMatrix(mol)
 
    if mol is not None:
       
        extended_adj_matrix = add_bond_nodes(adj_matrix_ethanol, mol)

    
    AllChem.EmbedMolecule(mol)

    if mol.GetNumConformers() == 0:
       
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))

    
    atom_positions = mol.GetConformer().GetPositions()

    adj_matrix_ethanol = Chem.GetAdjacencyMatrix(mol)

    
    distance_angle_matrix = calculate_all_bond_lengths_and_angles(atom_positions, adj_matrix_ethanol, mol)


    #return atoms_list,bonds_list,atoms_bonds_list,extended_adj_matrix,distance_matrix,angle_matrix
    
    return atoms_bonds_list,extended_adj_matrix,distance_angle_matrix

    #return atoms_bonds_list,extended_adj_matrix
