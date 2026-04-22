import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from rdkit import Chem
from pyscf import dft, gto
from rdkit.Chem import AllChem

HARTREE_TO_EV = 27.211386245988

@dataclass
class QCSpec:
    xc: str = "B3LYP"
    basis: str = "6-31g(d,p)" # "6-31g(d,p)"
    charge: int = 0
    multiplicity: int = 1
    conv_tol: float = 1e-8
    max_cycle: int = 200
    grid_level: int = 3
    level_shift: float = 0.0
    damp: float = 0.0
    use_newton: bool = False

def eh_to_ev(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(x) * HARTREE_TO_EV

def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None
        return xf
    except Exception:
        return None

def np_to_py(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, dict):
        return {k: np_to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [np_to_py(v) for v in obj]
    return obj

def get_optimizer() -> Tuple[str, Any]:
    try:
        from pyscf.geomopt.geometric_solver import optimize
        return "geomeTRIC", optimize
    except Exception:
        from pyscf.geomopt.berny_solver import optimize
        return "PyBerny", optimize

def smiles_to_rdkit(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    return mol

def embed_3d_geometry(smiles: str, random_seed: int = 42) -> Tuple[Chem.Mol, List[Tuple[str, Tuple[float, float, float]]]]:
    mol = smiles_to_rdkit(smiles)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.useRandomCoords = False

    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f"3D embedding failed for SMILES: {smiles}")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        if ff is not None:
            ff.Minimize(maxIts=2000)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000)

    conf = mol.GetConformer()
    atoms: List[Tuple[str, Tuple[float, float, float]]] = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append((atom.GetSymbol(), (float(pos.x), float(pos.y), float(pos.z))))
    return mol, atoms

def atom_list_to_string(atoms: Sequence[Tuple[str, Tuple[float, float, float]]]) -> str:
    lines = []
    for sym, (x, y, z) in atoms:
        lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    return "\n".join(lines)

def make_mol(
    atoms: Sequence[Tuple[str, Tuple[float, float, float]]],
    basis: str,
    charge: int,
    multiplicity: int,
    verbose: int = 0,
) -> gto.Mole:
    if multiplicity < 1:
        raise ValueError(f"Multiplicity must be >= 1, got {multiplicity}")
    spin = multiplicity - 1  # Nalpha - Nbeta
    mol = gto.M(
        atom=atom_list_to_string(atoms),
        basis=basis,
        charge=charge,
        spin=spin,
        unit="Angstrom",
        verbose=verbose,
    )
    return mol

def make_mf(mol: gto.Mole, spec: QCSpec):
    unrestricted = (spec.multiplicity != 1) or (mol.spin != 0)
    if unrestricted:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = spec.xc
    mf.conv_tol = spec.conv_tol
    mf.max_cycle = spec.max_cycle
    mf.level_shift = spec.level_shift
    mf.damp = spec.damp
    mf.grids.level = spec.grid_level
    if spec.use_newton: # No density fitting / no RI on purpose.
        mf = mf.newton()
        mf.max_cycle = spec.max_cycle
    return mf

def coords_from_mol(mol: gto.Mole) -> List[Tuple[str, Tuple[float, float, float]]]:
    symbols = mol.elements
    coords = mol.atom_coords(unit="Angstrom")
    return [(sym, (float(x), float(y), float(z))) for sym, (x, y, z) in zip(symbols, coords)]

def run_single_point(
    atoms: Sequence[Tuple[str, Tuple[float, float, float]]],
    spec: QCSpec,
) -> Dict[str, Any]:
    mol = make_mol(atoms, basis=spec.basis, charge=spec.charge, multiplicity=spec.multiplicity)
    mf = make_mf(mol, spec)
    e_tot = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError("SCF did not converge")

    props = collect_state_properties(mf)
    props["energy_hartree"] = e_tot
    props["energy_ev"] = eh_to_ev(e_tot)
    props["geometry_atoms"] = coords_from_mol(mol)
    return props

def optimize_structure(
    atoms: Sequence[Tuple[str, Tuple[float, float, float]]],
    spec: QCSpec,
    maxsteps: int = 100,
) -> Dict[str, Any]:
    mol = make_mol(atoms, basis=spec.basis, charge=spec.charge, multiplicity=spec.multiplicity)
    mf0 = make_mf(mol, spec)
    e0 = float(mf0.kernel())
    if not mf0.converged:
        raise RuntimeError("Initial SCF before optimization did not converge")

    optimizer_name, optimize = get_optimizer()
    mol_eq = optimize(mf0, maxsteps=maxsteps)

    # Re-run SCF on optimized geometry to get consistent final energy/properties.
    final_atoms = coords_from_mol(mol_eq)
    mol_final = make_mol(final_atoms, basis=spec.basis, charge=spec.charge, multiplicity=spec.multiplicity)
    mf_final = make_mf(mol_final, spec)
    e_final = float(mf_final.kernel())
    if not mf_final.converged:
        raise RuntimeError("Final SCF after optimization did not converge")

    result = collect_state_properties(mf_final)
    result.update(
        {
            "optimizer_backend": optimizer_name,
            "initial_energy_hartree": e0,
            "initial_energy_ev": eh_to_ev(e0),
            "energy_hartree": e_final,
            "energy_ev": eh_to_ev(e_final),
            "geometry_atoms": final_atoms,
        }
    )
    return result

def collect_state_properties(mf) -> Dict[str, Any]:
    mol = mf.mol
    data: Dict[str, Any] = {
        "scf_converged": bool(getattr(mf, "converged", False)),
        "total_charge": int(mol.charge),
        "multiplicity": int(mol.multiplicity),
        "n_atoms": int(mol.natm),
        "n_basis_functions": int(mol.nao_nr()),
    }
    try: # Dipole moment
        dip = np.array(mf.dip_moment(unit="Debye", verbose=0), dtype=float)
        data["dipole_debye_xyz"] = dip.tolist()
        data["dipole_debye_norm"] = float(np.linalg.norm(dip))
    except Exception:
        data["dipole_debye_xyz"] = None
        data["dipole_debye_norm"] = None
    try: # Mulliken charges
        _, chg = mf.mulliken_pop(verbose=0)
        data["mulliken_charges"] = [safe_float(x) for x in chg]
    except Exception:
        data["mulliken_charges"] = None
    try: # Spin square for unrestricted states
        s2, mult = mf.spin_square()
        data["s2"] = safe_float(s2)
        data["spin_square_multiplicity_estimate"] = safe_float(mult)
    except Exception:
        data["s2"] = None
        data["spin_square_multiplicity_estimate"] = None
    data.update(extract_frontier_orbitals(mf))
    return data

def extract_frontier_orbitals(mf) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ

    def frontier_from_arrays(energies, occs, prefix: str):
        energies = np.asarray(energies, dtype=float).reshape(-1)
        occs = np.asarray(occs, dtype=float).reshape(-1)

        # Numerical tolerance avoids brittleness with tiny fractional occupations.
        occ_idx = np.flatnonzero(occs > 1e-8)
        vir_idx = np.flatnonzero(occs <= 1e-8)

        homo = float(energies[occ_idx[-1]]) if occ_idx.size else None
        lumo = float(energies[vir_idx[0]]) if vir_idx.size else None

        out[f"{prefix}_homo_hartree"] = homo
        out[f"{prefix}_homo_ev"] = eh_to_ev(homo)
        out[f"{prefix}_lumo_hartree"] = lumo
        out[f"{prefix}_lumo_ev"] = eh_to_ev(lumo)
        if homo is not None and lumo is not None:
            gap = lumo - homo
            out[f"{prefix}_gap_hartree"] = gap
            out[f"{prefix}_gap_ev"] = eh_to_ev(gap)
        else:
            out[f"{prefix}_gap_hartree"] = None
            out[f"{prefix}_gap_ev"] = None

    mo_energy_arr = np.asarray(mo_energy, dtype=object)
    mo_occ_arr = np.asarray(mo_occ, dtype=object)

    # UKS/UHF can arrive either as ndarray with shape (2, nmo) or as tuple/list.
    if mo_occ_arr.ndim == 2 and mo_occ_arr.shape[0] == 2:
        frontier_from_arrays(mo_energy_arr[0], mo_occ_arr[0], "alpha")
        frontier_from_arrays(mo_energy_arr[1], mo_occ_arr[1], "beta")
    elif isinstance(mo_energy, (tuple, list)) and len(mo_energy) == 2:
        frontier_from_arrays(mo_energy[0], mo_occ[0], "alpha")
        frontier_from_arrays(mo_energy[1], mo_occ[1], "beta")
    else:
        frontier_from_arrays(mo_energy, mo_occ, "restricted")
    return out

def derive_molecule_defaults(record: Dict[str, Any], rdkit_mol_with_h: Chem.Mol) -> Tuple[int, int, int, int]:
    default_neutral_charge = int(Chem.GetFormalCharge(Chem.RemoveHs(rdkit_mol_with_h)))
    neutral_charge = int(record.get("neutral_charge", record.get("charge", default_neutral_charge)))
    neutral_mult = int(record.get("neutral_multiplicity", record.get("multiplicity", 1)))
    cation_charge = int(record.get("cation_charge", neutral_charge + 1))
    cation_mult = int(record.get("cation_multiplicity", 2))
    return neutral_charge, neutral_mult, cation_charge, cation_mult

# main functions
def hre_from_smiles(smiles: str,
                            xc: str = "B3LYP",
                            basis: str = "6-31g(d,p)",
                            max_steps: int = 100):
    rdkit_mol_h, init_atoms = embed_3d_geometry(smiles)
    neutral_charge, neutral_mult, cation_charge, cation_mult = derive_molecule_defaults({}, rdkit_mol_h)
    neutral_spec = QCSpec(xc=xc, basis=basis, charge=neutral_charge, multiplicity=neutral_mult)
    cation_spec = QCSpec(xc=xc, basis=basis, charge=cation_charge, multiplicity=cation_mult)
    neutral_opt = optimize_structure(init_atoms, neutral_spec, maxsteps=max_steps)
    cation_opt = optimize_structure(init_atoms, cation_spec, maxsteps=max_steps)
    e_cn = run_single_point(neutral_opt["geometry_atoms"], cation_spec) 
    e_nc = run_single_point(cation_opt["geometry_atoms"], neutral_spec)  
    e_nn = float(neutral_opt["energy_hartree"])  
    e_cc = float(cation_opt["energy_hartree"]) 
    e_c_n = float(e_cn["energy_hartree"])      
    e_n_c = float(e_nc["energy_hartree"])       
    lambda_relax_cation = e_c_n - e_cc
    lambda_relax_neutral = e_n_c - e_nn
    lambda_h = lambda_relax_cation + lambda_relax_neutral
    return eh_to_ev(lambda_h)

def ere_from_smiles(smiles: str,
                            xc: str = "B3LYP",
                            basis: str = "6-31g(d,p)",
                            max_steps: int = 100):
    rdkit_mol_h, init_atoms = embed_3d_geometry(smiles)
    neutral_charge, neutral_mult, _, _ = derive_molecule_defaults({}, rdkit_mol_h)
    anion_charge = neutral_charge - 1
    anion_mult = 2  
    neutral_spec = QCSpec(xc=xc, basis=basis, charge=neutral_charge, multiplicity=neutral_mult)
    anion_spec  = QCSpec(xc=xc, basis=basis, charge=anion_charge, multiplicity=anion_mult)
    neutral_opt = optimize_structure(init_atoms, neutral_spec, maxsteps=max_steps)
    anion_opt   = optimize_structure(init_atoms, anion_spec, maxsteps=max_steps)
    e_an = run_single_point(neutral_opt["geometry_atoms"], anion_spec) 
    e_na = run_single_point(anion_opt["geometry_atoms"], neutral_spec)
    e_nn = float(neutral_opt["energy_hartree"])  
    e_aa = float(anion_opt["energy_hartree"])   
    e_a_n = float(e_an["energy_hartree"])       
    e_n_a = float(e_na["energy_hartree"])       
    lambda_e = (e_a_n - e_aa) + (e_n_a - e_nn)
    return eh_to_ev(lambda_e)

def energies_from_smiles(smiles: str,
                        xc: str = "B3LYP",
                        basis: str = "6-31g(d,p)",
                        max_steps: int = 100):
    """calculate both HRE and ERE"""
    # Embed 3D geometry
    rdkit_mol_h, init_atoms = embed_3d_geometry(smiles)

    # Default charges/multiplicities
    neutral_charge, neutral_mult, cation_charge, cation_mult = derive_molecule_defaults({}, rdkit_mol_h)
    anion_charge = neutral_charge - 1
    anion_mult = 2

    # Build QC specs
    neutral_spec = QCSpec(xc=xc, basis=basis, charge=neutral_charge, multiplicity=neutral_mult)
    cation_spec  = QCSpec(xc=xc, basis=basis, charge=cation_charge, multiplicity=cation_mult)
    anion_spec   = QCSpec(xc=xc, basis=basis, charge=anion_charge, multiplicity=anion_mult)

    # Optimize geometries
    neutral_opt = optimize_structure(init_atoms, neutral_spec, maxsteps=max_steps)
    cation_opt  = optimize_structure(init_atoms, cation_spec, maxsteps=max_steps)
    anion_opt   = optimize_structure(init_atoms, anion_spec, maxsteps=max_steps)

    # --- Hole reorganization energy ---
    e_cn = run_single_point(neutral_opt["geometry_atoms"], cation_spec)  # E_C^N
    e_nc = run_single_point(cation_opt["geometry_atoms"], neutral_spec)  # E_N^C
    e_nn = float(neutral_opt["energy_hartree"])  # E_N^N
    e_cc = float(cation_opt["energy_hartree"])   # E_C^C
    e_c_n = float(e_cn["energy_hartree"])        # E_C^N
    e_n_c = float(e_nc["energy_hartree"])        # E_N^C
    lambda_h = (e_c_n - e_cc) + (e_n_c - e_nn)

    # --- Electron reorganization energy ---
    e_an = run_single_point(neutral_opt["geometry_atoms"], anion_spec)  # E_A^N
    e_na = run_single_point(anion_opt["geometry_atoms"], neutral_spec)  # E_N^A
    e_aa = float(anion_opt["energy_hartree"])       # E_A^A
    e_a_n = float(e_an["energy_hartree"])           # E_A^N
    e_n_a = float(e_na["energy_hartree"])           # E_N^A
    lambda_e = (e_a_n - e_aa) + (e_n_a - e_nn)
    return eh_to_ev(lambda_h), eh_to_ev(lambda_e)
