'''The overall objective of this file is to input a csv file (containing binding affinities(pIC50), and read the 3D structures of the pockets(PDB)),
and the chemical ligands(SDF), and merge them into 3D structural complexes. We convert them into graphs(features), and save them into the LMDB database.
Comments were made with assistance from Gemini, verified by Lakshana & Ben
'''
import gc

import pandas as pd
import os
from data.data_utils import pmap
from feats.gnina_types.obabel_api import merge_sdf_pdb_by_rdkit
from feats.third_rd_lib import load_by_rdkit, sdf_load, sdf_load_full, load_sdf_at_once
import glob
from data.data_stucture.lmdb_dataset import Subset
from data.dataset.common_dataset import Dataset



class BindingData(Dataset):
    """Dataset class for managing and processing protein-ligand binding data.

    This class handles data filtering, dataset splitting, and transformation of 
    structural files (PDB and SDF) into optimized sub-datasets for machine 
    learning training, validation, and inference.
    """
    def _filter_df(self, threshold, df, filter_type='normal'):
        """Filters the dataset rows based on binding affinity (pIC50) thresholds.

        Eliminates low-confidence or uninformative binding entries based on the chosen
        filtering scheme. It also strips out minimal near-zero binding affinities to 
        reduce background noise.

        Args:
            threshold (float): The minimum acceptable pIC50 value.
            df (pd.DataFrame): The input dataset DataFrame to be filtered.
            filter_type (str, optional): The strategy mode used for filtering 
                ('normal', 'hinge', or 'full'). Defaults to 'normal'.

        Returns:
            pd.DataFrame: A re-indexed DataFrame containing only the rows that met 
                the filtering criteria.
        """
        filter_msg = f"filter by (pic50) > {threshold}"
        df = df.astype({self.label_key: float})
        filter_bf = len(df)
        ######
        # Filter Normal
        condition = (df[self.label_key] > threshold)
        if filter_type == 'normal' or filter_type == 'hinge':
            condition = condition & (df[self.label_key].abs() > 0.1)
            filter_msg += "& abs(pic50) > 0.1"
        df = df[condition].reset_index(drop=True)
        print(f"[Bindingdata] {filter_msg}, {filter_bf} -> {len(df)}")
        #######
        # Filter Mer-Type Ligand pairs
        # filter_bf = len(df)
        # df = df.astype({'Molecule ID': str})
        # df = df[~df['Molecule ID'].str.contains('mer')]  # no need for filter
        # df = df.reset_index(drop=True)
        # print(f"[Bindingdata] Filter Mer types Ligand, {filter_bf} -> {len(df)}")
        return df

    @staticmethod
    def get_threshold(filter_type):
        """Determines the numerical bounding threshold corresponding to a specific filter type.

        Args:
            filter_type (str): The configuration mode name ('normal', 'hinge', or 'full').

        Returns:
            float: The lower limit numeric cut-off value used for binding affinity filtering.
                Returns negative infinity (`float('-inf')`) for unconstrained modes.
        """
        threshold = 0.1
        if filter_type == 'hinge':
            threshold = float('-inf')
        elif filter_type == 'full':
            threshold = float('-inf')
        return threshold

    def split(self, data, df):
        """Splits the input dataset into train, validation, and test subsets.

        Note:
            The original temporal and core-set exclusion splitting sequence has been 
            bypassed. If the runtime args designate `inference=True`, standard train/val 
            allocations are skipped, and the entire sample size is funneled exclusively 
            into the test set.

        Args:
            data (Any): The underlying structural dataset/graph storage collection.
            df (pd.DataFrame): The accompanying metadata DataFrame indexing the targets.

        Returns:
            tuple: A three-element tuple containing:
                - train (Subset): The designated training partition.
                - valid (Subset): The designated validation partition.
                - test (Subset): The designated testing partition.
        """
        def read_strip(f):
            return [x.strip() for x in open(f).readlines()]

        def exclude_coreset(train_pdbs, coresetlist_f):
            coreset = read_strip(coresetlist_f)
            train_pdbs = [x for x in train_pdbs if x not in coreset]
            return train_pdbs

        # commented this section out because we already received preprocessed code(train/test) from Hamza
        # read splits files
        # data_dir = os.path.dirname(self.args['data_path']) + f'/{self.args["split_folder"]}'
        # train_pdbs = read_strip(f'{data_dir}/timesplit_no_lig_overlap_train')
        # train_pdbs = read_strip(f'{data_dir}/new_timesplit_no_lig_overlap_train.txt')
        # Dataset that is not to be addressed
        # train_pdbs = exclude_coreset(train_pdbs, f'{data_dir}/coresetlist')
        # valid_pdbs = read_strip(f'{data_dir}/timesplit_no_lig_overlap_val')
        # Not to be considered for training
        # test_pdbs = read_strip(f'{data_dir}/timesplit_test')
        # train_indices = df[df['Target'].isin(train_pdbs)].index.tolist()
        # valid_indices = df[df['Target'].isin(valid_pdbs)].index.tolist()
        # test_indices = df[df['Target'].isin(test_pdbs)].index.tolist()
        if self.args['inference']:
            print("[BindingData]-Split: Inference Mode, set all data to testset.")
            train_indices = []
            valid_indices = []
            test_indices = df.index.tolist()
        train, valid, test = Subset(data, train_indices),  Subset(data, valid_indices), Subset(data, test_indices)
        print(f"train:{len(train)}/valid:{len(valid)}/test:{len(test)}")
        return train, valid, test

    def load_data(self, cache_path):
        """Attempt to load a previously cached dataset from local storage path.

        Args:
            cache_path (str): The absolute or relative path to the cached data file.

        Returns:
            Any: The populated dataset object if cache loading is enabled and the 
                 file exists; otherwise, None.
        """
        # load cached file
        if self.args['reload'] and self._check_exists(cache_path):
            wdata = self._get_dataset(cache_path)
            return wdata
        return None

    def __init__(self, args, istrain=True, n_jobs=1):
        """Initializes the BindingData pipeline and builds or loads the core dataset.

        Configures runtime target keys, filters raw dataframes, evaluates local cache 
        signatures, triggers structural molecular preprocessing, and coordinates split allocations.

        Note:
            Standard dataset split routines have been modified. When `istrain=True`, 
            the pipeline overrides the external split files and enforces an automated 
            70% training and 15% validation slice across all parsed samples.

        Args:
            args (dict): Dictionary mapping containing infrastructure configurations, 
                file directories, and parameter hyper-variables.
            istrain (bool, optional): Toggles training-mode data filtering and validation 
                splitting. Defaults to True.
            n_jobs (int, optional): Number of parallel CPU workers allocated for 
                multiprocess data generation. Defaults to 1.
        """
        self.set_type = 'lmdb'
        print(f"[Bindingdata] Dataloader:{self.set_type}")
        self.label_key = 'pIC50'
        self.target_key = 'Target'  # Uniprot
        self.dataset_type = args['filter_type']
        self.args = args
        threshold = self.get_threshold(args['filter_type'])
        # cached name
        cached_folder = f"{args['work_path']}/tmp_beta"
        cache_path = (f"{cached_folder}/"
                      f"{os.path.basename(args['data_path'])}-{args['method']}-{args['filter_type']}")
        cache_path = cache_path + '-uff-' if self.args['use_ff_ligands'] else cache_path
        cache_path = cache_path + '-affinity-' if self.args['affinity_pre'] else cache_path
        if args['affinity_pre']:
            print("[Bindingdata] preprocess for affinity mode.")
        ###
        # load previous exists csv
        data = self.load_data(cache_path)
        if data is None:
            df = pd.read_csv(args['data_path'])
            df = self._filter_df(threshold, df, self.dataset_type) if istrain else df
            # Create Complex Data
            data = self._pre_complex(df, cache_path, n_jobs)
        #
        # if istrain:
            # we already have splitted data, so we are now going to feed it via Hamza's data
            # self.datasets = self.split(data, self.df)
            # Force the dataset to just use the data it parsed without doing standard splits
        # We wrap it in a Subset-like structure if the training loop expects a tuple,
        # otherwise we just assign it.
        
        # If the code downstream expects a tuple of (train, val, test) when istrain=True,
        # we will just assign the whole dataset to train, and leave val/test empty.
        if istrain:
             from data.data_stucture.lmdb_dataset import Subset
             all_indices = list(range(len(data)))
             
             # Option A: If you want 15% of your train.csv to be used as Validation
             val_split_idx = int(len(data) * 0.85)
             train_subset = Subset(data, all_indices[:val_split_idx])
             val_subset = Subset(data, all_indices[val_split_idx:])
             test_subset = Subset(data, []) # We will test separately
             
             self.datasets = (train_subset, val_subset, test_subset)
             print(f"[BindingData] Custom Split: train:{len(train_subset)}/valid:{len(val_subset)}/test:{len(test_subset)}")
             
        else:
            self.datasets = data
        print(f"[Bindingdata] Total Samples:{len(self.df)}")

    @staticmethod
    def _docking_pose_handler(df, top=1):
        """Expands the working dataframe to account for multiple docked pose variants per target.

        Args:
            df (pd.DataFrame): The input dataframe containing molecule target configurations.
            top (int, optional): The number of top-ranked structural poses to consider. 
                Defaults to 1.

        Returns:
            tuple: A two-element tuple containing:
                - poses (List[int]): Index numbers tracking individual structural conformation rankings.
                - df (pd.DataFrame): The structurally replicated and expanded target dataframe.
        """
        if top > 1:
            print(f"[Bindingdata] Using Top{top} Poses..")
            df_index = df.index.repeat(top)
            t = df.loc[df_index].reset_index(drop=True)
            d = [i for i in range(top)] * (len(df_index) // top)
            t['poses'] = d
            poses = t['poses'].tolist()
            df = t
        else:
            poses = [0] * len(df)
        return poses, df

    @staticmethod
    def _check_pocket(pocket_folder, t):
        """Verifies the existence of a target protein pocket PDB file inside a given folder.

        Args:
            pocket_folder (str): The workspace subdirectory containing protein structure structures.
            t (str): The unique target identifier string (e.g., UniProt ID).

        Returns:
            str: The verified or fallback destination path string pointing to the pocket PDB.
        """
        pocket_f = pocket_folder + f'/{t}_pocket.pdb'
        # directly find
        if os.path.exists(pocket_f):
            return pocket_f
        # still not found
        return pocket_f

    def _read_pocket(self, uni_targets, n_jobs=1):
        """Parse pocket structural files into RDKit molecule definitions.

        Args:
            uni_targets (list): Cleaned collection of unique protein target identifiers.
            n_jobs (int, optional): CPU compute core count assigned to parallel file loading. 
                Defaults to 1.

        Returns:
            dict: Map linking target string IDs to initialized RDKit Mol protein objects.
        """
        # Load pocket pdb to mol objects
        print("Loading Pocket...")
        pocket_folder = f"{self.args['work_path']}/{self.args['pocket_path']}"
        count = 0
        mol_pdb_file = [self._check_pocket(pocket_folder, t) for t in uni_targets]
        pocket_mols = {}
        tmp_mols = pmap(load_by_rdkit, mol_pdb_file, format='pdb', n_jobs=n_jobs)
        for i, t in enumerate(uni_targets):
            if tmp_mols[i]:
                pocket_mols[t] = tmp_mols[i]
                count += 1
        print(f"[Bindingdata] Pocket count:{count}/{len(uni_targets)}\n" + '=' * 100)
        return pocket_mols

    def _get_merge_pairs(self, target, mids, ids, uniprot, docked, pocket_mols, n_jobs):
        """Assembles paired array combinations of pockets and ligands to run 3D composite merging.

        Note:
            Includes an internal exception tracker to prevent localized parsing 
            failures from stalling the pipeline, inserting null placeholder targets instead.

        Args:
            target (list): Target identifiers assigned to each dataset sample row.
            mids (list): Molecule IDs tracking chemical variants.
            ids (list): Int indices identifying targeted docking poses.
            uniprot (list): Accompanying UniProt identifier annotations.
            docked (dict): Lookup map holding extracted RDKit ligand structures.
            pocket_mols (dict): Lookup map holding parsed RDKit pocket structures.
            n_jobs (int): Thread workers handling matrix processing operations.

        Returns:
            list: Parsed structural feature matrices tracking protein-ligand contact properties.
        """
        print("[Bindingdata] Merging Pocket->Sdf, calculating PLIP.")
        complex_pairs = []
        merge_count = 0
        N = len(ids)
        for i in range(N):
            t = target[i]
            id = f'{t}_{mids[i]}_{ids[i]}'
            try:
                complex_pairs.append([t, uniprot[i], docked[id], pocket_mols[t]])
                merge_count += 1
            except Exception as e:
                print(f"Merge failed for ID {id}: {e}") #added debugging print for pinpointing 
                complex_pairs.append([None, None, None, None])
                pass
        # pmap - parallel mapping: use diff CPUs to speed up the calculations
        complex_mols = pmap(merge_sdf_pdb_by_rdkit, complex_pairs, n_jobs=n_jobs)
        print(f"[Bindingdata] Finished Merging Pocket+Ligand, {merge_count}/{N}\n" + '=' * 100)
        return complex_mols

    def _process_uff_ligands(self, complex_mols, targets, ids):
        """Integrates Universal Force Field (UFF) relaxed ligand data straight into structural targets.

        Loads bulk macro-SDF datasets iteratively grouped by primary targets to avoid 
        memory depletion bottlenecks.

        Args:
            complex_mols (list): Pre-assembled structural feature dictionaries.
            targets (list): Sequence list tracking structural target assignments per sample.
            ids (list): Index parameters tracking chosen configuration ranks.

        Returns:
            list: Updated compound configuration dictionaries containing appended force field metrics.
        """
        def find_uff_ligand(uff_folder, pdb):
            # using pdb to find
            f = glob.glob(f'{uff_folder}/{pdb}_uff.sdf')
            if len(f):
                return f[0]
            return ''  # no UFF file

        uff_folder = f"{self.args['work_path']}/{self.args['use_ff_ligands']}"
        print(f"[Bindingdata] Using FF ligands now..->{uff_folder}")
        # uff sdf may be extremely large, so we load each sdf by target name first
        uni_target = list(set(targets))
        t2sdf = {}
        for t in uni_target:
            t2sdf[t] = sdf_load_full(find_uff_ligand(uff_folder, t))
        data = []
        for i, t in enumerate(targets):
            uff_mol = t2sdf[t][ids[i]]
            if uff_mol:
                tmp_dict = complex_mols[i]
                tmp_dict['uff_ligand'] = uff_mol
                tmp_dict['mid'] = ids[i]
                tmp_dict['uff_as_ligand'] = self.args['uff_as_ligand']
                data.append(tmp_dict)
            else:
                data.append({'complex_str': None})
                print(t, end='_uff,')
        print(f"[Bindingdata] finished loading UFF.\n" + '=' * 100 + '\n')
        return data

    def _pre_complex(self, df, cache_path, n_jobs=1):
        """Orchestrates the entire structural assembly pipeline for protein-ligand systems.

        Extracts pocket coordinates, isolates ligand conformers from target SDF files, 
        resolves docking coordinate alignments, strips memory arrays, overlays UFF properties, 
        and calculates graph feature vectors before serializing outputs to database cache files.

        Args:
            df (pd.DataFrame): Screened source spreadsheet indexing target parameters.
            cache_path (str): Intended target path where structural metrics are cached.
            n_jobs (int, optional): Processing cores assigned to concurrent data parsing. 
                Defaults to 1.

        Returns:
            Any: An optimized database connection collection (e.g., LMDB dataset) populated 
                 with valid 3D graphs.

        Raises:
            ValueError: Critical safeguard exception triggered if 100% of spatial 
                merging tasks fail processing, indicating paths or target files are missing.
        """
        print(f"[Bindingdata] Making complex data {len(df)}, Method:{self.args['method']}")
        # 1. read pocket
        uniprot = df['Uniprot'].tolist() if 'Uniprot' in df else df['Target'].tolist()
        target = df['Target'].tolist()
        uni_targets = df['Target'].unique().tolist()
        pocket_mols = self._read_pocket(uni_targets, n_jobs=n_jobs)
        # 2. read sdf from file (and extract ligand from the file)
        print(f"[Bindingdata] Loading Ligands SDF...<-{self.args['ligand_folder']}")
        # 3. Assemble target configurations for each SDF payload.
        # Maps rows to: [absolute_sdf_filepath, molecule_id, pose_rank_idx, target_name]
        msg = 'Molecule ID' if self.args['use_mid'] else 'pose_rank'
        print(f'# [Bindingdata] using {msg} to select mol from sdfs.')
        if 'pose_rank' not in df:
            print("[Bindingdata] PoseRank Not exists in csv, using the first pose.")
            df['pose_rank'] = 0
        df = df.astype(dtype={'pose_rank': int})
        ids = df['pose_rank'].tolist()
        # 3.1 user may use mid (mol id) - compile ligand metadata and file routing rules
        if self.args['use_mid']:
            mids = df['Molecule ID'].tolist()
        else:
        #ensure that the length of the mids is the same len of the df - or else we'll get an index error(mismatched data structure len)
            mids = [''] * len(df)
        #format for where to find the sdf files
        sdfs_data = [
            [f'{self.args["work_path"]}/{self.args["ligand_folder"]}/{row["Target"]}_docked.sdf',
             mids[i], ids[i], row['Target']]
            for i, row in
            df.iterrows()]
        #extract ligand atomic coords. from the target files
        if self.args['vs']:
            #virtual screening mode: Read unified dataset sequentially in a single I/O sweep
            res = load_sdf_at_once(sdfs_data, use_mid=self.args['use_mid'])
        else:
            # Multi-Target Mode: Read independent files concurrently across distinct CPU threads
            res = pmap(sdf_load, sdfs_data, use_mid=self.args['use_mid'], n_jobs=n_jobs)

        # 3.2 Build Quick-Lookup Ligand Dictionary Mapping ID to RDKit Objects
        docked = {}
        for r in res:
            if r:
                docked[r[1]] = r[0]
        print(f"[Bindingdata] Loaded num of ligands: {len(docked)}/{len(res)}\n" + '=' * 100)
        # 3. Merge pocket & ligand docking pose pdb together
        complex_mols = self._get_merge_pairs(target, mids, ids, uniprot, docked, pocket_mols, n_jobs)
        # clear cached
        del docked
        del res
        del pocket_mols
        gc.collect()
        # 3.5 use-ff-ligands, wrap intra-D into complex_mols
        if self.args['use_ff_ligands'] and self.args['use_ff_ligands'] != "''":
            uff_ids = df['uff_pose_rank'].tolist() if 'uff_pose_rank' in df else [0] * len(df)
            complex_mols = self._process_uff_ligands(complex_mols, target, uff_ids)
        # 4. make features by using complex mol
        complex_data = pmap(self.args['complex_to_data'],
                            complex_mols,
                            node_featurizer=self.args['node_featurizer'],
                            edge_featurizer=self.args['edge_featurizer'],
                            args=self.args,
                            n_jobs=n_jobs)
        # 5. Keep only valid molecules
        valid_ids = []
        data = []
        for i, g in enumerate(complex_data):
            if g is not None:
                valid_ids.append(i)
                data.append(g)
            else:
                print(target[i], end=',')
        print(f"@ [Bindingdata] Valid Samples:{len(valid_ids)} / {len(complex_data)}")
        df = df.iloc[valid_ids].reset_index(drop=True)
        # write the label together
        if self.label_key not in df:  # avoid inference failure
            df[self.label_key] = 0.
        data = self._assign_label2x(data, df)
        # added checkpoint safeguard for ensuring that data is being loaded properly
        if len(data) == 0:
            raise ValueError("All samples failed processing! 'data' list is empty. Check your paths and IDs.")
        self._write_dataset(data, df, cache_path)
        wdata = self._get_dataset(cache_path)
        print('\n' + '+' * 100)
        return wdata
