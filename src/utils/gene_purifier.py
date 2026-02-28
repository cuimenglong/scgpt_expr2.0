"""
Gene Purifier - Converts Ensembl IDs to Gene Symbols
Based on previous/3_protein/gene_purifier.py
"""

import gseapy as gp
import pandas as pd
import mygene
import re


def convert_with_gseapy(adata):
    """
    Convert Ensembl IDs to Gene Symbols using GSEAPY (BioMart) + MyGene
    
    This function:
    1. Removes version numbers from Ensembl IDs
    2. Uses GSEAPY/BioMart for primary conversion
    3. Uses MyGene for backup conversion
    4. Handles duplicates by making names unique
    
    Args:
        adata: Scanpy AnnData object
        
    Returns:
        adata with gene names converted to symbols
    """
    print("Starting multi-stage gene ID conversion (GSEAPY + MyGene)...")
    
    # 1. Preprocess: remove version numbers (e.g., ENSG00000145075.12 -> ENSG00000145075)
    original_ids = adata.var_names.tolist()
    clean_ids = [re.sub(r'\.\d+$', '', str(g)) for g in original_ids]
    
    # Record mapping dictionary
    mapping = {}
    
    # 2. First round: Try GSEAPY (BioMart)
    ensembl_ids = [g for g in clean_ids if g.startswith("ENSG")]
    if ensembl_ids:
        try:
            print(f"Querying BioMart for {len(ensembl_ids)} genes...")
            bm = gp.Biomart()
            results = bm.query(dataset='hsapiens_gene_ensembl',
                               attributes=['ensembl_gene_id', 'external_gene_name'],
                               filters={'ensembl_gene_id': ensembl_ids})
            
            # Filter empty values
            results = results.dropna(subset=['external_gene_name'])
            # Build mapping table
            for _, row in results.iterrows():
                mapping[row['ensembl_gene_id']] = row['external_gene_name']
        except Exception as e:
            print(f"GSEAPY query failed, using backup engine: {e}")
    
    # 3. Second round: Fill missing with MyGene
    missing_ids = [g for g in ensembl_ids if g not in mapping]
    if missing_ids:
        try:
            print(f"Using MyGene to query remaining {len(missing_ids)} genes...")
            mg = mygene.MyGeneInfo()
            res = mg.querymany(missing_ids, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)
            
            for item in res:
                if 'symbol' in item:
                    mapping[item['query']] = item['symbol']
        except Exception as e:
            print(f"MyGene backup query failed: {e}")
    
    # 4. Apply mapping safely
    new_names = []
    success_count = 0
    
    for i in range(len(original_ids)):
        raw_id = original_ids[i]
        clean_id = clean_ids[i]
        
        # Priority: mapping table > clean_id > original_id
        target = mapping.get(clean_id, clean_id)
        
        # Ensure string and uppercase
        if pd.isna(target) or target is None:
            final_name = str(raw_id).upper()
        else:
            final_name = str(target).upper()
            if final_name.startswith("ENSG"):  # If still starts with ENSG, conversion failed
                pass
            else:
                success_count += 1
        
        new_names.append(final_name)
    
    # 5. Update adata
    adata.var['original_id'] = original_ids
    adata.var_names = new_names
    
    # Handle duplicates (common after Symbol conversion)
    if adata.var_names.duplicated().any():
        print("Detected duplicate Symbols, handling...")
        adata.var_names_make_unique()
    
    print(f"Conversion complete: Successfully converted {success_count} Ensembl IDs to Symbols")
    
    # Print unconverted genes as hints
    still_ensg = [n for n in adata.var_names if n.startswith("ENSG")]
    if still_ensg:
        print(f"Note: {len(still_ensg)} genes could not be converted and will use IDs.")
        print(f"Examples: {still_ensg[:5]}")
    
    return adata
