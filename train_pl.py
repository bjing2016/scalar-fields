from utils.parsing import parse_train_args
args = parse_train_args()
import wandb, os
from datasets.pdbbind import PDBBind
from model.wrapper import ModelWrapper
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import pytorch_lightning as pl
from utils.logging import get_logger

logger = get_logger(__name__)

if args.wandb:
    wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        settings=wandb.Settings(start_method="fork"),
        project="fft",
        name=args.run_name,
        config=args,
    )

def pyg_collate_fn(data_list):
    batch = Batch.from_data_list(data_list)
    return batch.to_dict()

def main():
    train_dataset = PDBBind(args, split_path=args.train_split_path)
    val_dataset = PDBBind(args, split_path=args.val_split_path)
                
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=pyg_collate_fn, 
        num_workers=args.num_workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=pyg_collate_fn,
        num_workers=args.num_workers,
    )

    model = ModelWrapper(args)

    trainer = pl.Trainer(
        default_root_dir=os.environ["MODEL_DIR"],
        accelerator="gpu",
        max_epochs=args.epochs,
        limit_train_batches=args.limit_batches or 1.0,
        limit_val_batches=args.limit_batches or 1.0,
        num_sanity_val_steps=0,
        enable_progress_bar=not args.wandb,
        callbacks=[ModelCheckpoint(save_top_k=-1)],
        gradient_clip_val=1,
        accumulate_grad_batches=args.accumulate_grad,
        check_val_every_n_epoch=args.val_freq,
    )
    
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.ckpt)
    

if __name__ == "__main__":
    main()
