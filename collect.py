import os
import numpy as np
import asyncio
import aiohttp
import aiofiles
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import logging
from pathlib import Path
import shutil
from Bio.PDB import PDBParser, PDBList
from Bio.PDB.Polypeptide import is_aa
from skimage.transform import resize

# ========== CONFIG ==========
NUM_PROTEINS = 50000
BATCH_SIZE = 500
DOWNLOAD_WORKERS = min(64, mp.cpu_count() * 4)
PROCESS_WORKERS = mp.cpu_count()
TEMP_DIR = "./temp_pdb"
MAPS_DIR = "./distance_maps"
MAP_COMPRESSION = True
MAX_RETRIES = 3
FRAGMENT_LENGTH = 40
OUTPUT_SIZE = 64
# ============================

# ========== URLS ==========
PDB_INDEX_URL = "https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx"
PDB_DOWNLOAD_BASE_URL = "https://files.rcsb.org/download"
# ============================



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_processed_proteins():
    """Get a set of already processed protein IDs from MAPS_DIR."""
    processed_files = Path(MAPS_DIR).glob("map_*.npz")
    processed_ids = {file.stem.split('_')[1] for file in processed_files}
    return processed_ids


def load_structure(pdb_file, pdb_id):
    """Load protein structure from PDB file"""
    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_id, pdb_file)

def extract_fragments(structure, length):
    """Extract fragments of given length from structure"""
    fragments = []
    for model in structure:
        for chain in model:
            residues = [res for res in chain if is_aa(res, standard=True)]
            for i in range(len(residues) - length + 1):
                frag = residues[i:i + length]
                fragments.append(frag)
    return fragments

def preprocess_map(dmap):
    """Preprocess distance map to standard size"""
    try:
        # Resize to standard size
        processed = resize(dmap, (OUTPUT_SIZE, OUTPUT_SIZE), preserve_range=True)
        return processed.astype(np.float32)
    except Exception:
        return None

def compute_distance_map(pdb_file):
    """Compute distance map from PDB file"""
    try:
        # Load structure and get fragments 
        structure = load_structure(pdb_file, os.path.basename(pdb_file)[:4])
        fragments = extract_fragments(structure, FRAGMENT_LENGTH)
        
        if not fragments:
            return None
            
        # Get C-alpha atoms from first fragment
        ca_atoms = [res['CA'] for res in fragments[0] if 'CA' in res]
        if len(ca_atoms) != FRAGMENT_LENGTH:
            return None
            
        # Compute pairwise distances
        dmap = np.zeros((FRAGMENT_LENGTH, FRAGMENT_LENGTH))
        for i, atom1 in enumerate(ca_atoms):
            for j, atom2 in enumerate(ca_atoms):
                # Calculate actual Euclidean distance between atoms
                dmap[i,j] = np.sqrt(np.sum((atom1.get_coord() - atom2.get_coord())**2))
                
        return dmap
    except Exception as e:
        logger.error(f"Error computing distance map: {e}")
        return None



def cleanup_temp_files():
    """Remove temporary directory and its contents"""
    import shutil
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        logger.info(f"Cleaned up temporary directory: {TEMP_DIR}")



async def download_single(session, pdb_id, semaphore):
    """Asynchronously download a single PDB file"""
    url = f"{PDB_DOWNLOAD_BASE_URL}/{pdb_id}.pdb"
    filepath = Path(TEMP_DIR) / f"{pdb_id}.pdb"
    
    async with semaphore:
        for retry in range(MAX_RETRIES):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        async with aiofiles.open(filepath, 'wb') as f:
                            await f.write(content)
                        return pdb_id, True
                    return pdb_id, False
            except Exception as e:
                if retry == MAX_RETRIES - 1:
                    logger.error(f"Failed to download {pdb_id}: {e}")
                    return pdb_id, False
                await asyncio.sleep(1)


async def download_batch(pdb_ids):
    """Download a batch of PDB files concurrently"""
    semaphore = asyncio.Semaphore(DOWNLOAD_WORKERS)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [download_single(session, pdb_id, semaphore) for pdb_id in pdb_ids]
        results = await asyncio.gather(*tasks)
    
    return [pdb_id for pdb_id, success in results if success]


def process_protein(pdb_file):
    """Process a single protein file"""
    try:
        structure = load_structure(pdb_file, os.path.basename(pdb_file)[:4])
        fragments = extract_fragments(structure, FRAGMENT_LENGTH)
        
        if not fragments:
            return None
        
        distance_maps = []
        for fragment in fragments:
            ca_atoms = [res['CA'] for res in fragment if 'CA' in res]
            if len(ca_atoms) != FRAGMENT_LENGTH:
                continue
            
            # Compute pairwise distances
            dmap = np.zeros((FRAGMENT_LENGTH, FRAGMENT_LENGTH))
            for i, atom1 in enumerate(ca_atoms):
                for j, atom2 in enumerate(ca_atoms):
                    dmap[i, j] = np.sqrt(np.sum((atom1.get_coord() - atom2.get_coord())**2))
            
            # Preprocess and append the distance map
            processed_map = preprocess_map(dmap)
            if processed_map is not None:
                distance_maps.append(processed_map)
        
        return distance_maps
    except Exception as e:
        logger.error(f"Error processing {pdb_file}: {e}")
        return None

        
async def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)
    total_size = 0
    successful_maps = 0

    try:
        # Get PDB IDs asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.get(PDB_INDEX_URL) as response:
                content = await response.text()

        # Parse PDB IDs
        all_pdb_ids = [line.split()[0] for line in content.split('\n')[2:] if line.strip()]

        # Get already processed protein IDs
        processed_ids = get_processed_proteins()

        # Filter out already processed IDs
        remaining_ids = [pdb_id for pdb_id in all_pdb_ids if pdb_id not in processed_ids]

        # Select new proteins to process
        selected_ids = np.random.choice(remaining_ids, NUM_PROTEINS, replace=False)

        # Process in batches
        for i in tqdm(range(0, len(selected_ids), BATCH_SIZE), desc="Processing batches"):
            batch_ids = selected_ids[i:i + BATCH_SIZE]
            
            # Download batch asynchronously
            successful_ids = await download_batch(batch_ids)
            
            # Process batch and save maps immediately
            pdb_files = [Path(TEMP_DIR) / f"{pdb_id}.pdb" for pdb_id in successful_ids]
            
            for idx, pdb_file in enumerate(pdb_files):
                if pdb_file.exists():
                    try:
                        # Process all fragments
                        distance_maps = process_protein(pdb_file)
                        if distance_maps:
                            for fragment_idx, dmap in enumerate(distance_maps):
                                map_path = Path(MAPS_DIR) / f"map_{successful_maps:05d}_{fragment_idx:02d}.npz"
                                np.savez_compressed(map_path, distance_map=dmap)
                                successful_maps += 1
                        pdb_file.unlink()
                    except Exception as e:
                        logger.error(f"Error processing {pdb_file}: {e}")
                        pdb_file.unlink()
                        continue        

            # Log progress and storage usage
            logger.info(f"Processed {successful_maps} maps. Total storage: {total_size / (1024*1024*1024):.2f} GB")

            # Optional: Clear memory
            import gc
            gc.collect()

    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        # Clean up temporary files
        cleanup_temp_files()
        logger.info(f"Final results: {successful_maps} maps processed. Total storage: {total_size / (1024*1024*1024):.2f} GB")
     
     
if __name__ == "__main__":
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        cleanup_temp_files()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        cleanup_temp_files()