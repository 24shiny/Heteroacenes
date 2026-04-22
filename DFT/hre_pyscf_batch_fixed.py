from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

from pyscf import dft, gto
from pyscf.hessian.thermo import harmonic_analysis, thermo


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


def read_smiles_table(path: str, smiles_col: str, id_col: Optional[str]) -> List[Dict[str, Any]]:
    in_path = Path(path)
    suffix = in_path.suffix.lower()

    if suffix in {".txt", ".smi"}:
        records: List[Dict[str, Any]] = []
        with open(in_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                smiles = parts[0]
                mol_id = parts[1] if len(parts) > 1 else f"mol_{i:05d}"
                records.append({"id": mol_id, "smiles": smiles})
        return records

    if suffix in {".json", ".jsonl"}:
        data: List[Dict[str, Any]] = []
        if suffix == ".jsonl":
            with open(in_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "smiles" not in row:
                        raise ValueError(f"JSONL line {i+1} does not contain 'smiles'")
                    data.append(row)
        else:
            payload = json.loads(in_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                data = payload
            else:
                raise ValueError("JSON input must be a list of objects")
        for i, row in enumerate(data):
            row.setdefault("id", row.get(id_col, f"mol_{i:05d}") if id_col else f"mol_{i:05d}")
        return data

    # default: CSV/TSV-like
    if suffix == ".tsv":
        df = pd.read_csv(in_path, sep="\t")
    else:
        df = pd.read_csv(in_path)

    if smiles_col not in df.columns:
        raise ValueError(f"Input file must contain a '{smiles_col}' column")

    if id_col is None or id_col not in df.columns:
        df["id"] = [f"mol_{i:05d}" for i in range(len(df))]
    else:
        df["id"] = df[id_col].astype(str)

    records = df.to_dict(orient="records")
    return records


def smiles_to_rdkit(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    return mol


def rdkit_descriptors(mol_no_h: Chem.Mol) -> Dict[str, Any]:
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol_no_h),
        "exact_mw": safe_float(Descriptors.ExactMolWt(mol_no_h)),
        "heavy_atom_count": int(mol_no_h.GetNumHeavyAtoms()),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol_no_h)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol_no_h)),
        "heteroatom_count": int(rdMolDescriptors.CalcNumHeteroatoms(mol_no_h)),
        "hbond_acceptor_count": int(rdMolDescriptors.CalcNumHBA(mol_no_h)),
        "hbond_donor_count": int(rdMolDescriptors.CalcNumHBD(mol_no_h)),
        "rotatable_bond_count": int(Lipinski.NumRotatableBonds(mol_no_h)),
        "tpsa": safe_float(rdMolDescriptors.CalcTPSA(mol_no_h)),
        "fraction_csp3": safe_float(rdMolDescriptors.CalcFractionCSP3(mol_no_h)),
        "logp": safe_float(Descriptors.MolLogP(mol_no_h)),
    }


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
    # No density fitting / no RI on purpose.
    if spec.use_newton:
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

    # Dipole moment
    try:
        dip = np.array(mf.dip_moment(unit="Debye", verbose=0), dtype=float)
        data["dipole_debye_xyz"] = dip.tolist()
        data["dipole_debye_norm"] = float(np.linalg.norm(dip))
    except Exception:
        data["dipole_debye_xyz"] = None
        data["dipole_debye_norm"] = None

    # Mulliken charges
    try:
        _, chg = mf.mulliken_pop(verbose=0)
        data["mulliken_charges"] = [safe_float(x) for x in chg]
    except Exception:
        data["mulliken_charges"] = None

    # Spin square for unrestricted states
    try:
        s2, mult = mf.spin_square()
        data["s2"] = safe_float(s2)
        data["spin_square_multiplicity_estimate"] = safe_float(mult)
    except Exception:
        data["s2"] = None
        data["spin_square_multiplicity_estimate"] = None

    # Frontier orbital energies
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


def run_frequency_analysis(mf, imag_threshold_cm1: float = 20.0) -> Dict[str, Any]:
    hess = mf.Hessian().kernel()
    vib = harmonic_analysis(mf.mol, hess, imaginary_freq=False)

    freq_wn = np.array(vib["freq_wavenumber"], dtype=float)
    imag = freq_wn[freq_wn < -abs(imag_threshold_cm1)]
    zpe = None
    thermo_data = None
    try:
        thermo_data = thermo(mf, vib["freq_au"], temperature=298.15, pressure=101325)
        if "ZPE" in thermo_data:
            zpe = float(thermo_data["ZPE"][0])
    except Exception:
        thermo_data = None

    return {
        "frequencies_cm1": [safe_float(x) for x in freq_wn.tolist()],
        "num_imag_freq": int(len(imag)),
        "most_negative_freq_cm1": safe_float(freq_wn.min()) if len(freq_wn) else None,
        "stable_minimum": bool(len(imag) == 0),
        "freq_imag_threshold_cm1": float(imag_threshold_cm1),
        "zpe_hartree": zpe,
        "zpe_ev": eh_to_ev(zpe) if zpe is not None else None,
        "thermo_298k": np_to_py(thermo_data) if thermo_data is not None else None,
    }


def kabsch_rmsd(
    atoms_a: Sequence[Tuple[str, Tuple[float, float, float]]],
    atoms_b: Sequence[Tuple[str, Tuple[float, float, float]]],
) -> Optional[float]:
    if len(atoms_a) != len(atoms_b):
        return None
    syms_a = [x[0] for x in atoms_a]
    syms_b = [x[0] for x in atoms_b]
    if syms_a != syms_b:
        return None

    pa = np.array([xyz for _, xyz in atoms_a], dtype=float)
    pb = np.array([xyz for _, xyz in atoms_b], dtype=float)

    pa_cent = pa.mean(axis=0)
    pb_cent = pb.mean(axis=0)
    xa = pa - pa_cent
    xb = pb - pb_cent

    h = xa.T @ xb
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    xa_rot = xa @ r
    diff = xa_rot - xb
    return float(np.sqrt((diff * diff).sum() / len(pa)))


def derive_molecule_defaults(record: Dict[str, Any], rdkit_mol_with_h: Chem.Mol) -> Tuple[int, int, int, int]:
    default_neutral_charge = int(Chem.GetFormalCharge(Chem.RemoveHs(rdkit_mol_with_h)))
    neutral_charge = int(record.get("neutral_charge", record.get("charge", default_neutral_charge)))
    neutral_mult = int(record.get("neutral_multiplicity", record.get("multiplicity", 1)))
    cation_charge = int(record.get("cation_charge", neutral_charge + 1))
    cation_mult = int(record.get("cation_multiplicity", 2))
    return neutral_charge, neutral_mult, cation_charge, cation_mult


def summarize_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    neutral = record.get("neutral", {})
    cation = record.get("cation", {})
    out = {
        "id": record.get("id"),
        "smiles": record.get("smiles"),
        "status": record.get("status"),
        "formula": record.get("formula"),
        "lambda_h_hartree": record.get("lambda_h_hartree"),
        "lambda_h_ev": record.get("lambda_h_ev"),
        "lambda_relax_cation_hartree": record.get("lambda_relax_cation_hartree"),
        "lambda_relax_cation_ev": record.get("lambda_relax_cation_ev"),
        "lambda_relax_neutral_hartree": record.get("lambda_relax_neutral_hartree"),
        "lambda_relax_neutral_ev": record.get("lambda_relax_neutral_ev"),
        "aip_hartree": record.get("aip_hartree"),
        "aip_ev": record.get("aip_ev"),
        "vip_hartree": record.get("vip_hartree"),
        "vip_ev": record.get("vip_ev"),
        "geometry_rmsd_angstrom": record.get("geometry_rmsd_angstrom"),
        "neutral_energy_hartree": neutral.get("energy_hartree"),
        "cation_energy_hartree": cation.get("energy_hartree"),
        "neutral_dipole_debye_norm": neutral.get("dipole_debye_norm"),
        "cation_dipole_debye_norm": cation.get("dipole_debye_norm"),
        "neutral_restricted_homo_ev": neutral.get("restricted_homo_ev"),
        "neutral_restricted_lumo_ev": neutral.get("restricted_lumo_ev"),
        "neutral_restricted_gap_ev": neutral.get("restricted_gap_ev"),
        "cation_alpha_homo_ev": cation.get("alpha_homo_ev"),
        "cation_alpha_lumo_ev": cation.get("alpha_lumo_ev"),
        "cation_beta_homo_ev": cation.get("beta_homo_ev"),
        "cation_beta_lumo_ev": cation.get("beta_lumo_ev"),
        "cation_s2": cation.get("s2"),
        "neutral_stable_minimum": (neutral.get("frequency") or {}).get("stable_minimum"),
        "cation_stable_minimum": (cation.get("frequency") or {}).get("stable_minimum"),
        "neutral_num_imag_freq": (neutral.get("frequency") or {}).get("num_imag_freq"),
        "cation_num_imag_freq": (cation.get("frequency") or {}).get("num_imag_freq"),
    }
    return out


def process_one_molecule(
    record: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    mol_id = str(record["id"])
    smiles = str(record["smiles"])

    rdkit_mol_h, init_atoms = embed_3d_geometry(smiles, random_seed=args.random_seed)
    rdkit_mol = Chem.RemoveHs(rdkit_mol_h)
    desc = rdkit_descriptors(rdkit_mol)

    neutral_charge, neutral_mult, cation_charge, cation_mult = derive_molecule_defaults(record, rdkit_mol_h)

    neutral_spec = QCSpec(
        xc=args.xc,
        basis=args.basis,
        charge=neutral_charge,
        multiplicity=neutral_mult,
        conv_tol=args.conv_tol,
        max_cycle=args.max_cycle,
        grid_level=args.grid_level,
        level_shift=args.level_shift,
        damp=args.damp,
        use_newton=args.newton,
    )
    cation_spec = QCSpec(
        xc=args.xc,
        basis=args.basis,
        charge=cation_charge,
        multiplicity=cation_mult,
        conv_tol=args.conv_tol,
        max_cycle=args.max_cycle,
        grid_level=args.grid_level,
        level_shift=args.level_shift,
        damp=args.damp,
        use_newton=args.newton,
    )

    neutral_opt = optimize_structure(init_atoms, neutral_spec, maxsteps=args.max_steps)
    cation_opt = optimize_structure(init_atoms, cation_spec, maxsteps=args.max_steps)

    # Cross single-points
    e_cn = run_single_point(neutral_opt["geometry_atoms"], cation_spec)  # E_C^N
    e_nc = run_single_point(cation_opt["geometry_atoms"], neutral_spec)  # E_N^C

    e_nn = float(neutral_opt["energy_hartree"])  # E_N^N
    e_cc = float(cation_opt["energy_hartree"])   # E_C^C
    e_c_n = float(e_cn["energy_hartree"])        # E_C^N
    e_n_c = float(e_nc["energy_hartree"])        # E_N^C

    lambda_relax_cation = e_c_n - e_cc
    lambda_relax_neutral = e_n_c - e_nn
    lambda_h = lambda_relax_cation + lambda_relax_neutral
    aip = e_cc - e_nn
    vip = e_c_n - e_nn

    if args.do_freq:
        neutral_freq_mol = make_mol(
            neutral_opt["geometry_atoms"], args.basis, neutral_charge, neutral_mult
        )
        neutral_freq_mf = make_mf(neutral_freq_mol, neutral_spec)
        neutral_freq_mf.kernel()
        neutral_opt["frequency"] = run_frequency_analysis(
            neutral_freq_mf, imag_threshold_cm1=args.imag_threshold_cm1
        )

        cation_freq_mol = make_mol(
            cation_opt["geometry_atoms"], args.basis, cation_charge, cation_mult
        )
        cation_freq_mf = make_mf(cation_freq_mol, cation_spec)
        cation_freq_mf.kernel()
        cation_opt["frequency"] = run_frequency_analysis(
            cation_freq_mf, imag_threshold_cm1=args.imag_threshold_cm1
        )

    result: Dict[str, Any] = {
        "id": mol_id,
        "smiles": smiles,
        "status": "ok",
        "formula": desc.get("formula"),
        **desc,
        "neutral_charge": neutral_charge,
        "neutral_multiplicity": neutral_mult,
        "cation_charge": cation_charge,
        "cation_multiplicity": cation_mult,
        "lambda_relax_cation_hartree": lambda_relax_cation,
        "lambda_relax_cation_ev": eh_to_ev(lambda_relax_cation),
        "lambda_relax_neutral_hartree": lambda_relax_neutral,
        "lambda_relax_neutral_ev": eh_to_ev(lambda_relax_neutral),
        "lambda_h_hartree": lambda_h,
        "lambda_h_ev": eh_to_ev(lambda_h),
        "aip_hartree": aip,
        "aip_ev": eh_to_ev(aip),
        "vip_hartree": vip,
        "vip_ev": eh_to_ev(vip),
        "geometry_rmsd_angstrom": kabsch_rmsd(
            neutral_opt["geometry_atoms"], cation_opt["geometry_atoms"]
        ),
        "energies_hartree": {
            "E_NN": e_nn,
            "E_CC": e_cc,
            "E_CN": e_c_n,
            "E_NC": e_n_c,
        },
        "neutral": neutral_opt,
        "cation": cation_opt,
        "cross_single_points": {
            "cation_on_neutral_geom": e_cn,
            "neutral_on_cation_geom": e_nc,
        },
        "method": {
            "program": "PySCF",
            "xc": args.xc,
            "basis": args.basis,
            "optimizer": get_optimizer()[0],
            "density_fitting": False,
            "ri": False,
        },
    }
    return np_to_py(result)


def write_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch PySCF hole reorganization energy workflow from a SMILES list."
    )
    p.add_argument("--input", required=True, help="Input file: csv/tsv/txt/smi/json/jsonl")
    p.add_argument("--smiles-col", default="smiles", help="SMILES column name for csv/tsv")
    p.add_argument("--id-col", default="id", help="ID column name for csv/tsv")
    p.add_argument("--output-dir", default="hre_pyscf_results", help="Output directory")
    p.add_argument("--xc", default="B3LYP", help="DFT functional, e.g. B3LYP, B3LYP, wb97x-d")
    p.add_argument("--basis", default="6-31g(d,p)", help="Basis set")
    p.add_argument("--conv-tol", type=float, default=1e-8, help="SCF convergence tolerance")
    p.add_argument("--max-cycle", type=int, default=200, help="SCF max cycle")
    p.add_argument("--grid-level", type=int, default=3, help="PySCF DFT grid level")
    p.add_argument("--level-shift", type=float, default=0.0, help="SCF level shift")
    p.add_argument("--damp", type=float, default=0.0, help="SCF damping")
    p.add_argument("--newton", action="store_true", help="Use PySCF Newton solver")
    p.add_argument("--max-steps", type=int, default=100, help="Geometry optimization max steps")
    p.add_argument("--do-freq", action="store_true", help="Run harmonic frequency analysis")
    p.add_argument(
        "--imag-threshold-cm1",
        type=float,
        default=20.0,
        help="Frequencies below -threshold are counted as imaginary modes",
    )
    p.add_argument("--random-seed", type=int, default=42, help="RDKit embedding seed")
    p.add_argument("--stop-on-error", action="store_true", help="Stop batch on first error")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = read_smiles_table(args.input, smiles_col=args.smiles_col, id_col=args.id_col)

    all_results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for i, record in enumerate(records, start=1):
        mol_id = str(record.get("id", f"mol_{i:05d}"))
        smiles = str(record.get("smiles", ""))
        print(f"[{i}/{len(records)}] Processing {mol_id}: {smiles}", flush=True)
        try:
            result = process_one_molecule(record, args)
        except Exception as exc:
            err = {
                "id": mol_id,
                "smiles": smiles,
                "status": "error",
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "method": {
                    "program": "PySCF",
                    "xc": args.xc,
                    "basis": args.basis,
                    "density_fitting": False,
                    "ri": False,
                },
            }
            result = err
            if args.stop_on_error:
                all_results.append(result)
                summary_rows.append({"id": mol_id, "smiles": smiles, "status": "error", "error_message": str(exc)})
                break

        all_results.append(result)
        summary_rows.append(summarize_for_csv(result) if result.get("status") == "ok" else {
            "id": result.get("id"),
            "smiles": result.get("smiles"),
            "status": "error",
            "error_message": result.get("error_message"),
        })

    jsonl_path = str(Path(args.output_dir) / "results.jsonl")
    csv_path = str(Path(args.output_dir) / "summary.csv")

    write_jsonl(all_results, jsonl_path)
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    meta = {
        "n_records": len(records),
        "n_ok": sum(1 for r in all_results if r.get("status") == "ok"),
        "n_error": sum(1 for r in all_results if r.get("status") != "ok"),
        "output_jsonl": jsonl_path,
        "output_csv": csv_path,
        "formula_hole_reorganization_energy": "lambda_h = (E_C^N - E_C^C) + (E_N^C - E_N^N)",
        "formula_aip": "AIP = E_C^C - E_N^N",
        "formula_vip": "VIP = E_C^N - E_N^N",
        "notes": [
            "No RI / no density fitting was used.",
            "Default assumption without multiplicity columns: neutral singlet, cation doublet.",
            "If --do-freq is used, stable_minimum=True means no imaginary mode below the chosen threshold.",
        ],
    }
    with open(Path(args.output_dir) / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Done. Summary CSV: {csv_path}")
    print(f"Done. Detailed JSONL: {jsonl_path}")
    return 0


# if __name__ == "__main__":
#     raise SystemExit(main())

def hre_from_smiles(smiles: str,
                            xc: str = "B3LYP",
                            basis: str = "6-31g(d,p)",
                            max_steps: int = 100):
    # Embed 3D geometry
    rdkit_mol_h, init_atoms = embed_3d_geometry(smiles)

    # Default charges/multiplicities
    neutral_charge, neutral_mult, cation_charge, cation_mult = derive_molecule_defaults({}, rdkit_mol_h)

    # Build QC specs
    neutral_spec = QCSpec(xc=xc, basis=basis, charge=neutral_charge, multiplicity=neutral_mult)
    cation_spec = QCSpec(xc=xc, basis=basis, charge=cation_charge, multiplicity=cation_mult)

    # Optimize geometries
    neutral_opt = optimize_structure(init_atoms, neutral_spec, maxsteps=max_steps)
    cation_opt = optimize_structure(init_atoms, cation_spec, maxsteps=max_steps)

    # Cross single-points
    e_cn = run_single_point(neutral_opt["geometry_atoms"], cation_spec) 
    e_nc = run_single_point(cation_opt["geometry_atoms"], neutral_spec)  

    e_nn = float(neutral_opt["energy_hartree"])  
    e_cc = float(cation_opt["energy_hartree"]) 
    e_c_n = float(e_cn["energy_hartree"])      
    e_n_c = float(e_nc["energy_hartree"])       

    # Compute properties
    lambda_relax_cation = e_c_n - e_cc
    lambda_relax_neutral = e_n_c - e_nn
    lambda_h = lambda_relax_cation + lambda_relax_neutral
    
    return eh_to_ev(lambda_h)

def ere_from_smiles(smiles: str,
                            xc: str = "B3LYP",
                            basis: str = "6-31g(d,p)",
                            max_steps: int = 100):
    # Embed 3D geometry
    rdkit_mol_h, init_atoms = embed_3d_geometry(smiles)

    # Default charges/multiplicities
    neutral_charge, neutral_mult, _, _ = derive_molecule_defaults({}, rdkit_mol_h)
    anion_charge = neutral_charge - 1
    anion_mult = 2  

    # Build QC specs
    neutral_spec = QCSpec(xc=xc, basis=basis, charge=neutral_charge, multiplicity=neutral_mult)
    anion_spec  = QCSpec(xc=xc, basis=basis, charge=anion_charge, multiplicity=anion_mult)

    # Optimize geometries
    neutral_opt = optimize_structure(init_atoms, neutral_spec, maxsteps=max_steps)
    anion_opt   = optimize_structure(init_atoms, anion_spec, maxsteps=max_steps)

    # Cross single-points for ERE
    e_an = run_single_point(neutral_opt["geometry_atoms"], anion_spec) 
    e_na = run_single_point(anion_opt["geometry_atoms"], neutral_spec)
    e_nn = float(neutral_opt["energy_hartree"])  
    e_aa = float(anion_opt["energy_hartree"])   
    e_a_n = float(e_an["energy_hartree"])       
    e_n_a = float(e_na["energy_hartree"])       

    # Electron reorganization energy
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
