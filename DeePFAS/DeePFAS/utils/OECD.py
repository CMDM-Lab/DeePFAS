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
            neighbors = [nbr for nbr in atom.GetNeighbors()]
            valid_f = 0
            for nbr in neighbors:
                if nbr.GetSymbol() == 'F' and len(nbr.GetNeighbors()) == 1:
                    valid_f += 1
            if valid_f in [2, 3] and not \
                any([bond.GetBondType() != Chem.BondType.SINGLE for bond in atom.GetBonds()]) and not \
                any([nbr.GetSymbol() in ['H', 'Br', 'I', 'Cl'] for nbr in neighbors]):
                        
                return True  # Perfluorinated methyl group found

    return False