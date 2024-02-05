import os
from timm.models import create_model, load_checkpoint

# create model
def get_sequencer_deep_lstm(model_dir):
    model = create_model(
        model_name='sequencer2d_l',
        pretrained=True,
        # checkpoint_path=os.path.join(model_dir, "sequencer2d_l.pth"),
        num_classes=1000,
        in_chans=3
    )

    return model
