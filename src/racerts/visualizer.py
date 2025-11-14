from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole  # noqa: F401 # type: ignore[attr-defined]
import py3Dmol
from typing import List, Tuple, Optional


def draw_multiple_confs(
    mol: Chem.Mol,
    max_number: int = 100,
    interval_size: int = 1,
    interval: Tuple[int, int] = (0, 1),
) -> None:
    """
    Uses Py3Dmol to visualize multiple conformers of a molecule.

    Args:
        mol (Chem.Mol): RDKit molecule object.
        max_number (int): Maximum number of conformers to display.
        interval_size (int): Interval size for stepping through conformers.
        interval (Tuple[int, int]): Tuple indicating the start and end of the conformer range.

    Returns:
        None
    """
    if max_number > 100:
        max_number = 100

    i = min(interval[0], mol.GetNumConformers())
    k = min(interval[1], mol.GetNumConformers())

    ids = [conf.GetId() for conf in mol.GetConformers()]

    p = py3Dmol.view(width=400, height=400)
    p.removeAllModels()

    for j in range(i, k, interval_size):
        IPythonConsole.addMolToView(mol, p, confId=ids[j])

    p.zoomTo()
    p.show()


def drawit(
    m: Chem.Mol,
    cids: Optional[List[int]] = None,
    p: Optional[py3Dmol.view] = None,
    removeHs: bool = True,
    colors: Tuple[str, ...] = (
        "cyanCarbon",
        "redCarbon",
        "blueCarbon",
        "magentaCarbon",
        "whiteCarbon",
        "purpleCarbon",
        "greenCarbon",
    ),
) -> None:
    """
    Visualizes molecules in 3D using Py3Dmol with options for coloring and removing hydrogens.

    Args:
        m (Chem.Mol): RDKit molecule object to visualize.
        cids (List[int], optional): List of conformer IDs to visualize. If None, visualizes all conformers.
        p (py3Dmol.view, optional): Py3Dmol viewer instance. If None, a new viewer will be created.
        removeHs (bool): If True, hydrogens will be removed from the molecule before visualization.
        colors (Tuple[str, ...]): Tuple of colors for the visualization, applied cyclically to models.

    Returns:
        None: Displays the molecule visualization.
    """
    if removeHs:
        m = Chem.RemoveHs(m)
    if p is None:
        p = py3Dmol.view(width=400, height=400)
    p.removeAllModels()

    if cids is None:
        cids = [conf.GetId() for conf in m.GetConformers()]

    if len(cids) > m.GetNumConformers():
        cids = cids[: len(cids)]

    for i, cid in enumerate(cids):
        IPythonConsole.addMolToView(m, p, confId=cid)
    for i, cid in enumerate(cids):
        p.setStyle(
            {
                "model": i,
            },
            {"stick": {"colorscheme": colors[i % len(colors)]}},
        )
    p.zoomTo()
    return p.show()
