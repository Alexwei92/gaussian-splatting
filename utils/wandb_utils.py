import wandb
import atexit
import torch


class WandbLogger:
    def __init__(self, project, **kwargs):
        try:
            self.run = wandb.init(
                project=project,
                dir="/tmp",
                **kwargs,
                settings=wandb.Settings(quiet=True),
            )
        except Exception as e:
            print(f"Failed to initialize WandB: {e}")
            self.run = None
        atexit.register(self.close)

    def add_image(self, tag, img_tensor, global_step=None):
        self.run.log(
            {tag: [wandb.Image(img_tensor)]},
            step=global_step if global_step is not None else None,
        )

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.run.log(
            {tag: scalar_value},
            step=global_step if global_step is not None else None,
        )

    def add_histogram(self, tag, values, global_step=None):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
            
        self.run.log(
            {tag: wandb.Histogram(values)},
            step=global_step if global_step is not None else None,
        )

    def close(self):
        if self.run:
            self.run.finish()
