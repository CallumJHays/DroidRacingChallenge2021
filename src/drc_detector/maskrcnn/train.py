# %%
"Transfer-training script adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-your-model"

import os
import sys
import time

import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor, Compose, Normalize, Resize

HERE = os.path.abspath(os.path.dirname(__file__))

# insert the path to resolve some import issues
sys.path.insert(0, f"{HERE}/pytorch_utils/detection")
from pytorch_utils.detection.engine import train_one_epoch, evaluate
from pytorch_utils.detection.utils import collate_fn

# replace the classifier with a new one, that has
# num_classes which is user-defined
COCO_DATASET_PATH = f"{HERE}/dataset"
CLASSES = ["__background__", "enemy", "obstacle", "yellow_line", "blue_line"]
HIDDEN_LAYER_SIZE = 256

MIN_TEST_SCORE = 0.4  # Objectness threshold
NUM_EPOCHS = 1
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_TEST_EXAMPLES = 5

# DRCDataset = CocoDetection
class DRCDataset(CocoDetection):
    """
    Override CocoDetection Dataset so that we can ignore images that don't have any labels.
    This way we can pick and choose which frames to label and train for in cvat.
    """

    def __init__(self, root: str, ann_path: str):
        super().__init__(
            root,
            ann_path,
            transforms=Compose([ToTensor(), Resize((240, 320)), Normalize()]),
        )

        labeled_img_ids = set()
        for ann in self.coco.anns.values():
            labeled_img_ids.add(ann["image_id"])

        self.ids = list(sorted(labeled_img_ids))

    def __getitem__(self, idx): 
         img, target = super(CocoDetection, self).__getitem__(idx) 
         image_id = self.ids[idx] 
         target = dict(image_id=image_id, annotations=target) 
         if self._transforms is not None: 
             img, target = self._transforms(img, target) 
         return img, target 

# %%
print(f"Loading dataset from path {COCO_DATASET_PATH}")

# Note this isn't the COCO dataset per-se; but it's in the same format.
dataset_whole = DRCDataset(
    f"{COCO_DATASET_PATH}/images",
    f"{COCO_DATASET_PATH}/annotations/instances_default.json",
)

print(f"Dataset has {len(dataset_whole)} examples (all labeled)")

# %%
print(f"Splitting dataset to get {NUM_TEST_EXAMPLES} test images")
indices = torch.randperm(len(dataset_whole)).tolist()
dataset_train = torch.utils.data.Subset(dataset_whole, indices[:-NUM_TEST_EXAMPLES])
dataset_test = torch.utils.data.Subset(dataset_whole, indices[-NUM_TEST_EXAMPLES:])

print(
    f"Training dataset has {len(dataset_train)} images. Test dataset has {len(dataset_test)} images."
)

# %%
print("Loading pretrained COCO model")

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# %%
print("Replacing output layers of model")

# replace the bounding box and mask predictors with untrained ones with our custom classes
fastrcnn_in = model.roi_heads.box_predictor.cls_score.in_features
maskrcnn_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.box_predictor = FastRCNNPredictor(fastrcnn_in, len(CLASSES))
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    maskrcnn_in, HIDDEN_LAYER_SIZE, len(CLASSES)
)

# %%
print("Preparing data loaders")

data_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
)

# %%
# train on the GPU or on the CPU, if a GPU is not available
if torch.cuda.is_available():
    print("Moving model to the GPU...")
    device = torch.device("cuda")
else:
    print("No GPU available. Using CPU")
    device = torch.device("cpu")
model.to(device)

# %%

print("Configuring training utilities")

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)


# %%
print(f"Beginning training for {NUM_EPOCHS} epochs")
start = time.time()
for epoch in range(NUM_EPOCHS):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    lr_scheduler.step()

    # evaluate on the test dataset
    # if epoch % 5 == 0 and epoch > 0:
    #     print('Evaluating...')
    #     evaluate(model, data_loader_test, device=device)

print(f"DONE TRAINING. took {time.time() - start} seconds")
# %%
print("Evaluating model")
model.eval()

# %%
weights_path = f"{HERE}/weights.pt"
print(f"Saving model weights to {weights_path}")

torch.save(model.state_dict(), weights_path)
print("Script complete!")
