import os
import copy
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy

from data.data_process import GraphDataModule
from utils.cluster import auto_configure_nccl
from utils.parser import get_args
from utils.train_utils import load_model, param_count, get_callbacks

# adjusted file paths
from data.sampler import per_target_balance_sampler, LISA_sampler, Fullsampler
from data.dataset.bindingdata import BindingData
from data.dataset.ppi_dataset import PPIData
import torch
from functools import partial
from torch.utils.data import distributed
from pytorch_lightning import LightningDataModule
from data.collator.inter_collate_fn import interformer_collate_fn
from data.collator.ppi_collate_fn import ppi_collate_fn, ppi_residue_collate_fn

print(f"# Torch Version:{torch.__version__}")


def main(args):
    # NCCL
    auto_configure_nccl()
    # Data Processing
    dm = GraphDataModule(args)
    # Model Setup
    model = load_model(args)
    param_count(model)
    # Start Training
    print('+' * 100)
    #
    print(f"# Precision:{args['precision']}f")
    # dataset_model
    folder_name = f"{os.path.basename(args['data_path'])[:-4]}_{args['model']}_{args['Code']}"
    print(f"#Folder_Name:{folder_name}")
    tb_logger = pl_loggers.TensorBoardLogger(f"{args['checkpoint']}/lightning_logs/", folder_name)
    trainer = pl.Trainer(
        devices='auto',
        precision=args['precision'],
        max_epochs=args['num_epochs'],
        log_every_n_steps=20,
        fast_dev_run=False,
        callbacks=get_callbacks(args),
        check_val_every_n_epoch=1,
        val_check_interval=1.,
        num_sanity_val_steps=0,  # num of batches in val, to check, -1 means the whole val
        accelerator='cuda',
        default_root_dir=args['checkpoint'],
        logger=tb_logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        use_distributed_sampler=False,  # it is important, make sure trainner not using their own sampler
        # reload_dataloaders_every_n_epochs=1,
        num_nodes=args['num_nodes'],
    )
    trainer.fit(model, datamodule=dm)
    Final Test
    # print(f"# Testing by:{trainer.checkpoint_callback.best_model_path}")
    # test_result = trainer.test(model, ckpt_path='best', datamodule=dm)
    # print(test_result)
    # print("+" * 100)
    # print("*********END of One Model*******")
    ## suggested modification on the training procedure for having already completed train/test split(Hamza)
    test_args = copy.deepcopy(args)
    test_args['data_path'] = 'data/proto_test_final.csv' 
    Create a new GraphDataModule for the test set
    test_dm = GraphDataModule(test_args)
    ------------------
    
    # Evaluate using the new test datamodule
    test_result = trainer.test(model, ckpt_path='best', datamodule=test_dm)
    return test_result


if __name__ == "__main__":
    args = get_args()
    # Main loop
    total = []
    for i in range(args['main_loop']):
        print(f"# Main Loop->{i}")
        total.append(main(args))
    print("DONE")
    dist.destroy_process_group()
