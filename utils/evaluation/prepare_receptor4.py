#!/usr/bin/env python3

import os
import sys
import getopt

from MolKit import Read
import MolKit.molecule
import MolKit.protein
from AutoDockTools.MoleculePreparation import AD4ReceptorPreparation

def usage():
    print("Usage: prepare_receptor4.py -r filename")
    print("\nDescription of command:")
    print("    -r   receptor_filename")
    print("         supported file types include pdb,mol2,pdbq,pdbqs,pdbqt, possibly pqr,cif")
    print("Optional parameters:")
    print("    [-v]  verbose output (default is minimal output)")
    print("    [-o pdbqt_filename]  (default is 'molecule_name.pdbqt')")
    print("    [-A]  type(s) of repairs to make:")
    print("         'bonds_hydrogens': build bonds and add hydrogens")
    print("         'bonds': build single bond from each atom with no bonds to its closest neighbor")
    print("         'hydrogens': add hydrogens")
    print("         'checkhydrogens': add hydrogens only if none exist")
    print("         'None': do not make any repairs")
    print("         (default is 'checkhydrogens')")
    print("    [-C]  preserve all input charges (default is addition of gasteiger charges)")
    print("    [-p]  preserve input charges on specific atom types, e.g., -p Zn -p Fe")
    print("    [-U]  cleanup type:")
    print("         'nphs': merge charges and remove non-polar hydrogens")
    print("         'lps': merge charges and remove lone pairs")
    print("         'waters': remove water residues")
    print("         'nonstdres': remove chains of non-standard residues")
    print("         'deleteAltB': remove XX@B atoms and rename XX@A atoms to XX")
    print("         (default is 'nphs_lps_waters_nonstdres')")
    print("    [-e]  delete every nonstd residue not in 20 standard amino acids")
    print("    [-M]  interactive (default is automatic)")

# Main
if __name__ == '__main__':
    try:
        opt_list, args = getopt.getopt(sys.argv[1:], 'r:vo:A:Cp:U:eM:')
    except getopt.GetoptError as msg:
        print(f'prepare_receptor4.py: {msg}')
        usage()
        sys.exit(2)

    receptor_filename = None
    verbose = False
    repairs = ''
    charges_to_add = 'gasteiger'
    preserve_charge_types = None
    cleanup = "nphs_lps_waters_nonstdres"
    outputfilename = None
    mode = 'automatic'
    delete_single_nonstd_residues = None

    for o, a in opt_list:
        if o == '-r':
            receptor_filename = a
            if verbose: print(f'set receptor_filename to {a}')
        elif o == '-v':
            verbose = True
            print('set verbose to True')
        elif o == '-o':
            outputfilename = a
            if verbose: print(f'set outputfilename to {a}')
        elif o == '-A':
            repairs = a
            if verbose: print(f'set repairs to {a}')
        elif o == '-C':
            charges_to_add = None
            if verbose: print('do not add charges')
        elif o == '-p':
            preserve_charge_types = a if not preserve_charge_types else preserve_charge_types + ',' + a
            if verbose: print(f'preserve initial charges on {preserve_charge_types}')
        elif o == '-U':
            cleanup = a
            if verbose: print(f'set cleanup to {a}')
        elif o == '-e':
            delete_single_nonstd_residues = True
            if verbose: print('set delete_single_nonstd_residues to True')
        elif o == '-M':
            mode = a
            if verbose: print(f'set mode to {a}')
        elif o == '-h':
            usage()
            sys.exit()

    if not receptor_filename:
        print('prepare_receptor4: receptor filename must be specified.')
        usage()
        sys.exit()

    mols = Read(receptor_filename)
    if verbose: print(f'read {receptor_filename}')
    mol = mols[0]
    preserved = {}

    if charges_to_add and preserve_charge_types:
        preserved_types = preserve_charge_types.split(',')
        if verbose: print(f"preserved_types = {preserved_types}")
        for t in preserved_types:
            if verbose: print(f'preserving charges on type -> {t}')
            if not len(t): continue
            ats = mol.allAtoms.get(lambda x: x.autodock_element == t)
            if verbose: print(f'preserving charges on {[a.name for a in ats]}')
            for a in ats:
                if a.chargeSet is not None:
                    preserved[a] = [a.chargeSet, a.charge]

    if len(mols) > 1:
        if verbose: print("more than one molecule in file")
        for m in mols[1:]:
            if len(m.allAtoms) > len(mol.allAtoms):
                mol = m
                if verbose: print(f"mol set to larger molecule with {len(mol.allAtoms)} atoms")

    mol.buildBondsByDistance()

    if verbose:
        print(f"setting up RPO with mode = {mode}, outputfilename = {outputfilename}")
        print(f"charges_to_add = {charges_to_add}")
        print(f"delete_single_nonstd_residues = {delete_single_nonstd_residues}")

    RPO = AD4ReceptorPreparation(
        mol, mode, repairs, charges_to_add, cleanup,
        outputfilename=outputfilename,
        preserved=preserved,
        delete_single_nonstd_residues=delete_single_nonstd_residues
    )

    if charges_to_add:
        for atom, chargeList in preserved.items():
            atom._charges[chargeList[0]] = chargeList[1]
            atom.chargeSet = chargeList[0]
