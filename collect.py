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

NUM_PROTEINS = 50000
BATCH_SIZE = 500
DOWNLOAD_WORKERS = min(64, mp.cpu_count() * 4)
PROCESS_WORKERS = mp.cpu_count()
TEMP_DIR = "./temp_pdb"
MAPS_DIR = "./new_distance_maps"
MAP_COMPRESSION = True
MAX_RETRIES = 3
FRAGMENT_LENGTH = 40
OUTPUT_SIZE = 64
PDB_INDEX_URL = "https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx"
PDB_DOWNLOAD_BASE_URL = "https://files.rcsb.org/download"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_processed_proteins():
    processed_files = Path(MAPS_DIR).glob("map_*.npz")
    processed_ids = {file.stem.split('_')[1] for file in processed_files}
    return processed_ids


def load_structure(pdb_file, pdb_id):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_id, pdb_file)

def extract_fragments(structure, length):
    fragments = []
    for model in structure:
        for chain in model:
            residues = [res for res in chain if is_aa(res, standard=True)]
            for i in range(len(residues) - length + 1):
                frag = residues[i:i + length]
                fragments.append(frag)
    return fragments

def preprocess_map(dmap):
    try:
        processed = resize(dmap, (OUTPUT_SIZE, OUTPUT_SIZE), preserve_range=True)
        return processed.astype(np.float32)
    except Exception:
        return None

def compute_distance_map(pdb_file):
    try:
        structure = load_structure(pdb_file, os.path.basename(pdb_file)[:4])
        fragments = extract_fragments(structure, FRAGMENT_LENGTH)
        
        if not fragments:
            return None
            
        ca_atoms = [res['CA'] for res in fragments[0] if 'CA' in res]
        if len(ca_atoms) != FRAGMENT_LENGTH:
            return None
            
        #calculating pairwise distances
        dmap = np.zeros((FRAGMENT_LENGTH, FRAGMENT_LENGTH))
        for i, atom1 in enumerate(ca_atoms):
            for j, atom2 in enumerate(ca_atoms):
                dmap[i,j] = np.sqrt(np.sum((atom1.get_coord() - atom2.get_coord())**2))
        return dmap
    except Exception as e:
        logger.error(f"Error computing distance map: {e}")
        return None



def cleanup_temp_files():
    import shutil
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        logger.info(f"Cleaned up temporary directory: {TEMP_DIR}")


async def download_single(session, pdb_id, semaphore):
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
    semaphore = asyncio.Semaphore(DOWNLOAD_WORKERS)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [download_single(session, pdb_id, semaphore) for pdb_id in pdb_ids]
        results = await asyncio.gather(*tasks)
    
    return [pdb_id for pdb_id, success in results if success]


def process_protein(pdb_file):
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
            
            dmap = np.zeros((FRAGMENT_LENGTH, FRAGMENT_LENGTH))
            for i, atom1 in enumerate(ca_atoms):
                for j, atom2 in enumerate(ca_atoms):
                    dmap[i, j] = np.sqrt(np.sum((atom1.get_coord() - atom2.get_coord())**2))
            
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
        async with aiohttp.ClientSession() as session:
            async with session.get(PDB_INDEX_URL) as response:
                content = await response.text()

        all_pdb_ids = [line.split()[0] for line in content.split('\n')[2:] if line.strip()]
        processed_ids = get_processed_proteins()
        remaining_ids = [pdb_id for pdb_id in all_pdb_ids if pdb_id not in processed_ids]
        selected_ids = np.random.choice(remaining_ids, NUM_PROTEINS, replace=False)

        for i in tqdm(range(0, len(selected_ids), BATCH_SIZE), desc="Processing batches"):
            batch_ids = selected_ids[i:i + BATCH_SIZE]
            successful_ids = await download_batch(batch_ids)
            pdb_files = [Path(TEMP_DIR) / f"{pdb_id}.pdb" for pdb_id in successful_ids]
            
            for idx, pdb_file in enumerate(pdb_files):
                if pdb_file.exists():
                    try:
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

            logger.info(f"Processed {successful_maps} maps. Total storage: {total_size / (1024*1024*1024):.2f} GB")
            import gc
            gc.collect()

    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
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
