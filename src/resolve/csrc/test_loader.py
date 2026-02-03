"""Test the fast CSV loader on actual data."""
import sys
sys.path.insert(0, '.')
from fast_loader import load_species_csv
import time

species_path = r'J:\Phd Local\Gilles_paper_resolve\data\ASAAS_SPECIES.csv'
print(f'Loading {species_path}...')
start = time.time()
result = load_species_csv(species_path, verbose=True)
elapsed = time.time() - start
print(f'Total time: {elapsed:.1f}s')
print(f'Records: {len(result["plot_indices"]):,}')
print(f'Plots: {len(result["plot_offsets"]) - 1:,}')
