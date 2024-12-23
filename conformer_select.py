from rdkit import Chem
from rdkit.Chem import AllChem

def has_3D_conformer(molecule):
    """Check if a molecule has a 3D conformer."""
    AllChem.EmbedMolecule(molecule)
    return molecule.GetNumConformers() > 0


def filter_molecules(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    header = lines[0].strip()

    smiles_list = [line.strip() for line in lines[1:] if line.strip()]

    # Convert SMILES to RDKit molecules
    molecules=[]
    for smiles in smiles_list:

       a=Chem.AddHs(Chem.MolFromSmiles(smiles))
       molecules.append(a)

    # Filter molecules with 3D conformers
    molecules_with_3D = []
    i=0
    for smiles in molecules:
        i=i+1
        print(i)
        print(Chem.MolToSmiles(Chem.RemoveHs(smiles)))
        if has_3D_conformer(smiles):
          molecules_with_3D.append(smiles)

    # Convert molecules back to SMILES
    filtered_smiles_list = [Chem.MolToSmiles(Chem.RemoveHs(mol)) for mol in molecules_with_3D]

    # Write results to a new file
    with open(output_file, 'w') as outfile:
        # Write the header
        outfile.write(header + '\n')
        # Write the filtered SMILES
        outfile.write('\n'.join(filtered_smiles_list))

input_filepath = 'data/chembl_145wan.txt'
output_filepath = 'data/chembl_conformer_select_145wan.txt'

filter_molecules(input_filepath, output_filepath)
