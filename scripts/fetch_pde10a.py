import requests, time, subprocess, os, warnings
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
parser = PDBParser()

for pdb_id in open('splits/pde10a'):
    
    pdb_id = pdb_id.strip()
    
    os.makedirs(f"data/pde10a/{pdb_id}", exist_ok=True)
    resp = requests.get(f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}').json()
    entity_ids = resp['rcsb_entry_container_identifiers']['non_polymer_entity_ids']
    found = False
    for entity_id in entity_ids:
        resp = requests.get(f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{entity_id}").json()
        if not 'rcsb_nonpolymer_entity_feature' in resp: continue
        for feature in resp['rcsb_nonpolymer_entity_feature']:
            if feature['type'] == 'SUBJECT_OF_INVESTIGATION':
                found = True
                break
        if found: break
    instance_ids = resp['rcsb_nonpolymer_entity_container_identifiers']['asym_ids']
    
    for instance_id in instance_ids:
        resp = requests.get(f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity_instance/{pdb_id}/{instance_id}").json()
        if resp['rcsb_target_neighbors'][0]['target_asym_id'] == 'A':
            break

    # these are mislabelled by depositors, must manually correct
    if pdb_id == '5sf0': instance_id = 'G' 
    if pdb_id == '5sef': instance_id = 'E'
        
    ligand_url = f"https://models.rcsb.org/v1/{pdb_id}/ligand?label_asym_id={instance_id}&encoding=SDF"
    #ligand_url = f"https://models.rcsb.org/v1/{pdb_id}/ligand?label_asym_id=G&encoding=SDF"
    protein_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(ligand_url)
    
    subprocess.run(['wget', protein_url, '-O', f"data/pde10a/{pdb_id}/{pdb_id}_protein.pdb"], stderr=subprocess.DEVNULL)
    subprocess.run(['wget', ligand_url, '-O', f"data/pde10a/{pdb_id}/{pdb_id}_ligand.sdf"], stderr=subprocess.DEVNULL)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        model = parser.get_structure('', f"data/pde10a/{pdb_id}/{pdb_id}_protein.pdb")[0]
    chain = model['A']
    for resi in list(chain):
        if (resi.id[0] != ' ') or ('CA' not in resi.child_dict):
            chain.detach_child(resi.id)
    
    pdbio = PDBIO()
    pdbio.set_structure(chain)
    pdbio.save(f"data/pde10a/{pdb_id}/{pdb_id}_protein_processed.pdb")
    
    time.sleep(1)
    