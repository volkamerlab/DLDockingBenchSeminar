import glob
import os
import sys
from tqdm import tqdm
import joblib
from rdkit import Chem
from oddt.toolkits.extras.rdkit import fixer
'''fixer: imported for using ExtractPocketAndLigand - extract ligand and protein pocket (w/ cutoff) '''


def create_ligand_residue(ligand):
    '''Modifies an RDKit molecule in-place to add or format PDB residue information.

    For structural files lacking formal PDB residue naming (like raw SDFs), this function 
    generates fallback residue metadata ('LIG') and labels the atoms as heteroatoms. 
    If residue metadata already exists, it flags them as non-heteroatoms to preserve 
    pocket-parsing compatibility.

    Args:
        ligand (rdkit.Chem.rdchem.Mol): The RDKit molecule object representing the ligand.

    Returns:
        nothing, the function already manipulates the ligand as desired.
    '''

    for atom in ligand.GetAtoms():
        res = atom.GetPDBResidueInfo()
        # sdf
        if res is None:
            symbol = atom.GetSymbol()
            lig_name = '  LIG' if len(symbol) == 1 else ' LIG'
            atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(' ' + symbol, residueName=lig_name))
            res = atom.GetPDBResidueInfo()
            res.SetResidueName(lig_name)
            res.SetIsHeteroAtom(True)
        else:
            if res.GetIsHeteroAtom():
                res.SetIsHeteroAtom(False)
    # debug
    # str_ = Chem.MolToPDBBlock(complex)
    pass


# def grep_unique_CCD(protein_f):
#   all = set()
#   for line in open(protein_f).readlines():
#     if line.startswith('HET'):
#       CCD = line[17:20]
#       all.add(CCD)
#   return list(all)

def grep_unique_CCD(protein_mol):
    all_CCD = set()
    for atom in protein_mol.GetAtoms():
        resn = atom.GetPDBResidueInfo()
        if resn.GetIsHeteroAtom():
            all_CCD.add(resn.GetResidueName().lstrip().rstrip())
    return all_CCD

def cut_digits(num):
    return float(format(num, '.3f'))

def grep_ref_CCD(ligand, pdb_f):
    xyz = list(ligand.GetConformer().GetPositions()[0])
    xyz[0],  xyz[1], xyz[2] = cut_digits(xyz[0]), cut_digits(xyz[1]), cut_digits(xyz[2])
    for line in open(pdb_f).readlines():
        if line.startswith('HETATM'):
            p_x, p_y, p_z = float(line[30:38]), float(line[39:46]), float(line[46:54])
            ccd = line[17:20]
            if xyz[0] == p_x and xyz[1] == p_y and xyz[2] == p_z:
                return ccd
    print('ccd not matching', pdb_f)
    # assert False
    return ""


def run_fn(pdb_f, rm_ccd=False):
    # New modification (to get pockets for multiple ccd)
    base = os.path.basename(pdb_f)
    pdbid = base.replace("_protein_refined.pdb", "")

    ligand_file = os.path.join(
        ligand_path,
        base.replace(
            "_protein_refined.pdb",
            "_ligand_refined.sdf"
         )
    )

    if not os.path.exists(ligand_file):
        print(f"Missing ligand: {ligand_file}")
        return

    ligand = Chem.SDMolSupplier(
        ligand_file,
        sanitize=False
    )[0]
    
    
    out_name = base.replace(
    "_protein_refined.pdb",
    "_pocket.pdb"
    )
    #original code:
    # pdbid = os.path.basename(pdb_f)[:4]
    #ligand = Chem.SDMolSupplier(f'{ligand_path}/{pdbid}_docked.sdf', sanitize=False)[0]
    #modified code:
    # matches = glob.glob(
    #      f"{ligand_path}/{pdbid}_*_ligand_refined.sdf"
    # )

    # if len(matches) == 0:
    #     print(f"No ligand found for {pdbid}")
    #     return

    # if len(matches) > 1:
    #     print(f"Multiple ligands found for {pdbid}: {matches}")
    #     return

    # ligand = Chem.SDMolSupplier(
    #     matches[0],
    #     sanitize=False
    # )[0]

    create_ligand_residue(ligand)
    # find out the reference CCD

    # protein
    pdb_mol = Chem.MolFromPDBFile(pdb_f, sanitize=False, removeHs=True)
    ###
    # Rm CCD
    ref_ccd = ''
    if rm_ccd:
        ref_ccd = grep_ref_CCD(ligand, pdb_f)
        if pdb_mol is None or ref_ccd == "":
            # skip
            print(f"Failed loading<-{pdb_f}")
            f = open(f'{protein_path}/output/failed', 'a')
            f.write(pdbid + '\n')
            f.close()
            return
    ccd_list = grep_unique_CCD(pdb_mol)
    ccd_list = list(ccd_list - {ref_ccd})  # exclude ref ccd
    #
    complex = Chem.rdmolops.CombineMols(ligand, pdb_mol)
    try:
        pocket, ligand = fixer.ExtractPocketAndLigand(complex, cutoff=10, append_residues=ccd_list,
                                                      expandResidues=True, ligand_residue='LIG')  # PDB standard
        '''Function extracting a ligand (the largest HETATM residue) and the protein
        pocket within certain cutoff. The selection of pocket atoms can be expanded
        to contain whole residues. The single atom HETATM residues are attributed
        to pocket (metals and waters)

        Parameters
        ----------
            mol: rdkit.Chem.rdchem.Mol
                Molecule with a protein ligand complex
            cutoff: float (default=12.)
                Distance cutoff for the pocket atoms
            expandResidues: bool (default=True)
                Expand selection to whole residues within cutoff.
            ligand_residue: string (default None)
                Residue name which explicitly pint to a ligand(s).
            ligand_residue_blacklist: array-like, optional (default None)
                List of residues to ignore during ligand lookup.
            append_residues: array-like, optional (default None)
                List of residues to append to pocket, even if they are HETATM, such
                as MSE, ATP, AMP, ADP, etc.

        Returns
        -------
            pocket: rdkit.Chem.rdchem.RWMol
                Pocket constructed of protein residues/atoms around ligand
            ligand: rdkit.Chem.rdchem.RWMol
                Largest HETATM residue contained in input molecule
        '''
    except Exception as e:
        print(f"Error<-{pdbid}, {e}")

    # w = Chem.PDBWriter(f'{protein_path}/output/{pdbid}_pocket.pdb')
    w = Chem.PDBWriter(
        f"{protein_path}/output/{out_name}"
    )
    w.write(pocket)
    w.close()

    return 0.


if __name__ == '__main__':
    print('[protein_root] [ligand_root] [0|1, rm_ccd or not]')
    protein_path = sys.argv[1]
    ligand_path = sys.argv[2]
    if len(sys.argv) > 3:
        rm_ccd = bool(int(sys.argv[3]))
    else:
        rm_ccd = True
    os.makedirs(protein_path + '/output', exist_ok=True)
    pdbs_f = list(
        glob.glob(f'{protein_path}/*.pdb'))  # please makesure you remove the reference ligand from this complex
    # pdbs_f = [x for x in pdbs_f if '3lqj' in x]  # debug
    joblib.Parallel(n_jobs=70)(joblib.delayed(run_fn)(x, rm_ccd=rm_ccd) for x in tqdm(pdbs_f))
