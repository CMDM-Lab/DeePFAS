from rdkit import Chem

def oecd_pfas(smiles):

    # follow revised OCED definition
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Error on parse SMILES string')

    mol = Chem.AddHs(mol)

    # Iterate through atoms and check if there are CF3 or CF2 groups
    for atom in mol.GetAtoms():
        # Check for perfluorinated methyl group (-CF3)
        if atom.GetSymbol() == 'C' and atom.GetDegree() >= 2:
            neighbors = [nbr.GetSymbol() for nbr in atom.GetNeighbors()]
            if neighbors.count('F') >= 2 and not any([nbr in ['H', 'Br', 'I', 'Cl'] for nbr in neighbors]):
                return True  # Perfluorinated methyl group found

    return False