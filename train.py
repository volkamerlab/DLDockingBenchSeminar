'''
Please note that in order to run this, you need:
1. A wandb account. We will send you an email to our team for you to view the plots
2. A .env file with the API key (in same dir as train.py) - we can upload the API key into the wandb secrets (on the wandb website) that you can 
    paste into your .env file.
'''
# imported libraries
from dotenv import load_dotenv
import wandb
# TODO: Lakshana will implement this into the Docker container for more detailed runs 
from datetime import datetime # for timestamping runs on wandb

import os
import copy
import torch
import pytorch_lightning as pl
from functools import partial
from torch.utils.data import distributed
from pytorch_lightning import LightningDataModule
import torch.distributed as dist
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy

# imported files from interformer
from data.data_process import GraphDataModule
from utils.cluster import auto_configure_nccl
from utils.parser import get_args
from utils.train_utils import load_model, param_count, get_callbacks
#sampling data and ppi
from data.sampler import per_target_balance_sampler, LISA_sampler, Fullsampler
from data.dataset.bindingdata import BindingData
from data.dataset.ppi_dataset import PPIData
from data.collator.inter_collate_fn import interformer_collate_fn
from data.collator.ppi_collate_fn import ppi_collate_fn, ppi_residue_collate_fn

# Switch sharing strategy from 'file_descriptor' to 'file_system'
torch.multiprocessing.set_sharing_strategy("file_system")

print(f"# Torch Version:{torch.__version__}")
load_dotenv() # load the API keys

def main(args):

    #enable for a100 (Thanks Hamza)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    #timestamp for wandb runs
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_run_name = f"New-Dock_{args['run_name']}_{current_time}"

    # NCCL - NVIDIA Collective Communications Library (NCCL)
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
    # tb_logger = pl_loggers.TensorBoardLogger(f"{args['checkpoint']}/lightning_logs/", folder_name)
    
    # Replacing tensorboard with wandb
    # wandb is preferred for us; so we don't have to run another script to extract the loss all the time
    # from the .out files - wandb will automate this process.
    # WandB Logger Configuration

    wandb_logger = pl_loggers.WandbLogger(
        entity="dl-docking",  # Set to your username/organization if needed
        project="Interformer",  # Your project name
        name=timestamped_run_name,
        log_model="all",  # Log model checkpoints - to obtain hparams.yaml file use "all" instead of "True"
        tags=["interformer", "docking"],
    )
    
    # Log hyperparameters on wandb (https://wandb.ai/cispa-phoenix/DL-Docking)
    wandb_logger = pl_loggers.WandbLogger(
        entity="dl-docking",  
        project="Interformer",  
        name=timestamped_run_name,
        log_model="all",  
        tags=["interformer", "docking"],
        config={                                
            'model': args['model'],
            'precision': args['precision'],
            'num_epochs': args['num_epochs'],
            'data_path': args['data_path'],
            'learning_rate': args.get('lr', 'N/A'),
            'batch_size': args.get('batch_size', 'N/A'),
        }
    )

    trainer = pl.Trainer(
        devices='auto',
        #created more options for precision (for a100)
        precision='16-mixed' if args['precision'] == 16 or args['precision'] == '16' else args['precision'],
        max_epochs=args['num_epochs'],
        log_every_n_steps=20,
        fast_dev_run=False,
        callbacks=get_callbacks(args),
        check_val_every_n_epoch=1,
        val_check_interval=1.,
        num_sanity_val_steps=0,  # num of batches in val, to check, -1 means the whole val
        accelerator='cuda',
        default_root_dir=args['checkpoint'],
        # logger=tb_logger, #moving to wandb and moving away from Tensorboard
        logger=wandb_logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        use_distributed_sampler=False,  # it is important, make sure trainner not using their own sampler
        # reload_dataloaders_every_n_epochs=1,
        num_nodes=args['num_nodes'],
        accumulate_grad_batches=5 # To make up for low batch size due to memory constrain (4*5=20) - energy model and (2*5) - pose affinity model
        # accumulate_grad_batches=2 # To make up for low batch size due to memory constrain and since 2GPU (5*2*2=20) - affinity-normal model 
    )
    trainer.fit(model, datamodule=dm)

    # Skipped for prototype training -- this was for the authors
    # Final Test
    # print(f"# Testing by:{trainer.checkpoint_callback.best_model_path}")
    # test_result = trainer.test(model, ckpt_path='best', datamodule=dm)
    # print(test_result)
    # print("+" * 100)
    # print("*********END of One Model*******")
    # suggested modification on the training procedure for having already completed train/test split(Hamza)
    # datapath and work path updated according to test data.
    # test_args = copy.deepcopy(args)
    # test_args['data_path'] = 'data/proto_test_final.csv'
    # test_args['work_path'] = 'data/proto_test' 
    # test_args['inference'] = True
    # # Create a new GraphDataModule for the test set
    # test_dm = GraphDataModule(test_args, istrain=False)
    
    # # # Evaluate using the new test datamodule
    # test_result = trainer.test(model, ckpt_path='best', datamodule=test_dm)
    # wandb_logger.finalize("success")
    # return test_result


if __name__ == "__main__":
    args = get_args()
    # Main loop
    total = []
    for i in range(args['main_loop']):
        print(f"# Main Loop->{i}")
        total.append(main(args))
    print("DONE")
    dist.destroy_process_group()
