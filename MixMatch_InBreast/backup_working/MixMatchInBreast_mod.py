from fastai.vision import *
from numbers import Integral
import seaborn as sns
#WHAT TRANSFORMATIONS ARE USED?
#IMAGE DIMENSIONS?

DEFAULT_PATH = "/media/Data/saul/Datasets/Inbreast_folder_per_class"
NUMBER_CLASSES = 6
NUMBER_LABELED_OBSERVATIONS = 100
BATCH_SIZE = 10
SIZE_IMAGE = 100
# Modified from
K = 2


class MultiTransformLabelList(LabelList):
    def __getitem__(self, idxs: Union[int, np.ndarray]) -> 'LabelList':
        "return a single (x, y) if `idxs` is an integer or a new `LabelList` object if `idxs` is a range."
        idxs = try_int(idxs)
        if isinstance(idxs, Integral):
            if self.item is None:
                x, y = self.x[idxs], self.y[idxs]
            else:
                x, y = self.item, 0
            if self.tfms or self.tfmargs:
                # I've changed this line to return a list of augmented images
                x = [x.apply_tfms(self.tfms, **self.tfmargs) for _ in range(K)]
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve': False})
            if y is None: y = 0
            return x, y
        else:
            return self.new(self.x[idxs], self.y[idxs])


# I'll also need to change the default collate function to accomodate multiple augments
def MixmatchCollate(batch):
    batch = to_data(batch)
    if isinstance(batch[0][0], list):
        batch = [[torch.stack(s[0]), s[1]] for s in batch]
    return torch.utils.data.dataloader.default_collate(batch)



#path = DEFAULT_PATH

class MixupLoss(nn.Module):
    def forward(self, preds, target, unsort=None, ramp=None, bs=None):
        if unsort is None:
            return F.cross_entropy(preds,target)
        preds = preds[unsort]
        preds_l = preds[:bs]
        preds_ul = preds[bs:]
        preds_l = torch.log_softmax(preds_l,dim=1)
        preds_ul = torch.softmax(preds_ul,dim=1)
        loss_x = -(preds_l * target[:bs]).sum(dim=1).mean()
        loss_u = F.mse_loss(preds_ul,target[bs:])
        self.loss_x = loss_x.item()
        self.loss_u = loss_u.item()
        return loss_x + 100 * ramp * loss_u

# Custom ImageList with filter function
class MixMatchImageList(ImageList):
    def filter_train(self, num_items, seed=2343):
        print("Filter train")
        print(self.items)
        # -2 for the MNIST structure!
        train_idxs = np.array([i for i, o in enumerate(self.items) if Path(o).parts[-3] != "test"])
        valid_idxs = np.array([i for i, o in enumerate(self.items) if Path(o).parts[-3] == "test"])
        np.random.seed(seed)
        # keep the number of items desired, 500 by default
        keep_idxs = np.random.choice(train_idxs, num_items, replace=False)
        print("Keep idxs")
        print(len(keep_idxs))
        print("Valid idxs")
        print(len(valid_idxs))
        print("train idxs")
        print(len(train_idxs))

        self.items = np.array([o for i, o in enumerate(self.items) if i in np.concatenate([keep_idxs, valid_idxs])])
        return self









def mixup(a_x,a_y,b_x,b_y,alpha=0.75):
    l = np.random.beta(alpha,alpha)
    l = max(l,1-l)
    x = l * a_x + (1-l) * b_x
    y = l* a_y + (1-l) * b_y
    return x,y

def sharpen(p,T=0.5):
    u = p ** (1/T)
    return u / u.sum(dim=1,keepdim=True)




class MixMatchTrainer(LearnerCallback):
    _order = -20

    def on_train_begin(self, **kwargs):
        self.l_dl = iter(data_labeled.train_dl)
        self.smoothL, self.smoothUL = SmoothenValue(0.98), SmoothenValue(0.98)
        self.recorder.add_metric_names(["l_loss", "ul_loss"])
        self.it = 0

    def on_batch_begin(self, train, last_input, last_target, **kwargs):
        if not train: return
        try:
            x_l, y_l = next(self.l_dl)
        except:
            self.l_dl = iter(data_labeled.train_dl)
            x_l, y_l = next(self.l_dl)

        x_ul = last_input

        with torch.no_grad():
            ul_labels = sharpen(
                torch.softmax(torch.stack([self.learn.model(x_ul[:, i]) for i in range(x_ul.shape[1])], dim=1),
                              dim=2).mean(dim=1))

        x_ul = torch.cat([x for x in x_ul])
        ul_labels = torch.cat([y.unsqueeze(0).expand(K, -1) for y in ul_labels])

        l_labels = torch.eye(data_labeled.c).cuda()[y_l]

        """
        print("X_l size")
        print(x_l.shape)
        print("x_ul size")
        print(x_ul.shape)
        print("L Labels")
        print(l_labels.shape)
        print("U Labels")
        print(ul_labels.shape)
        """
        w_x = torch.cat([x_l, x_ul])
        w_y = torch.cat([l_labels, ul_labels])
        idxs = torch.randperm(w_x.shape[0])

        mixed_input, mixed_target = mixup(w_x, w_y, w_x[idxs], w_y[idxs])
        bn_idxs = torch.randperm(mixed_input.shape[0])
        unsort = [0] * len(bn_idxs)
        for i, j in enumerate(bn_idxs): unsort[j] = i
        mixed_input = mixed_input[bn_idxs]

        ramp = self.it / 3000.0 if self.it < 3000 else 1.0
        return {"last_input": mixed_input, "last_target": (mixed_target, unsort, ramp, x_l.shape[0])}

    def on_batch_end(self, train, **kwargs):
        if not train: return
        self.smoothL.add_value(self.learn.loss_func.loss_x)
        self.smoothUL.add_value(self.learn.loss_func.loss_u)
        self.it += 1

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, [self.smoothL.smooth, self.smoothUL.smooth])

# print(data_labeled)

# Grab file path to cifar dataset. Will download data if not present
path = untar_data(URLs.CIFAR)
path = DEFAULT_PATH

meanDatasetComplete = [0.1779, 0.1779, 0.1779]
stdDatasetComplete =  [0.2539, 0.2539, 0.2539]
inbreast_stats = (meanDatasetComplete, stdDatasetComplete)


print("NORMALIZATION TYPE:")
print(cifar_stats)
print("NORMALIZATION TYPE INBREAST:")
print(inbreast_stats)

print("Loading data from: ", path)
# Create two databunch objects for the labeled and unlabled images. A fastai databunch is a container for train, validation, and
# test dataloaders which automatically processes transforms and puts the data on the gpu.
#https://docs.fast.ai/vision.transform.html
"""
Utility func to easily create a list of flip, rotate, zoom, warp, lighting transforms.

do_flip: if True, a random flip is applied with probability 0.5
flip_vert: requires do_flip=True. If True, the image can be flipped vertically or rotated by 90 degrees, otherwise only an horizontal flip is applied
max_rotate: if not None, a random rotation between -max_rotate and max_rotate degrees is applied with probability p_affine
max_zoom: if not 1. or less, a random zoom between 1. and max_zoom is applied with probability p_affine
max_lighting: if not None, a random lightning and contrast change controlled by max_lighting is applied with probability p_lighting
max_warp: if not None, a random symmetric warp of magnitude between -max_warp and maw_warp is applied with probability p_affine
p_affine: the probability that each affine transform and symmetric warp is applied
p_lighting: the probability that each lighting transform is applied
xtra_tfms: a list of additional transforms you would like to be applied
"""
data_labeled = (MixMatchImageList.from_folder(path)
                .filter_train(NUMBER_LABELED_OBSERVATIONS)  # Use 500 labeled images for traning
                .split_by_folder(valid="test")  # test on all 10000 images in test set
                .label_from_folder()
                .transform(get_transforms(max_zoom = 1, max_warp =None, p_affine = 0 ), size=SIZE_IMAGE)
                # On windows, must set num_workers=0. Otherwise, remove the argument for a potential performance improvement
                .databunch(bs=BATCH_SIZE, num_workers=10)
                .normalize(inbreast_stats))


#normalize_funcs(mean:FloatTensor, std:FloatTensor, do_x:bool=True, do_y:bool=False)
train_set = set(data_labeled.train_ds.x.items)
src = (ImageList.from_folder(path)
       .filter_by_func(lambda x: x not in train_set)
       .split_by_folder(valid="test"))
src.train._label_list = MultiTransformLabelList
#https://docs.fast.ai/vision.transform.html
data_unlabeled = (src.label_from_folder()
                  .transform(get_transforms(max_zoom=1, max_warp=None, p_affine=0), size=SIZE_IMAGE)
                  .databunch(bs=BATCH_SIZE, collate_fn=MixmatchCollate, num_workers=10)
                  .normalize(inbreast_stats))

# Databunch with all 50k images labeled, for baseline
data_full = (ImageList.from_folder(path)
             .split_by_folder(valid="test")
             .label_from_folder()
             .transform(get_transforms(max_zoom=1, max_warp=None, p_affine=0), size=SIZE_IMAGE)
             .databunch(bs=BATCH_SIZE, num_workers=10)
             .normalize(inbreast_stats))

#start_nf the initial number of features
"""
Wide ResNet with num_groups and a width of k.
Each group contains N blocks. start_nf the initial number of features. Dropout of drop_p is applied in between the two convolutions in each block. The expected input channel size is fixed at 3.
Structure: initial convolution -> num_groups x N blocks -> final layers of regularization and pooli
"""
model = models.WideResNet(num_groups=3,N=4,num_classes=NUMBER_CLASSES,k=2,start_nf=SIZE_IMAGE)

print("Training fully supervised model")
learnFS = Learner(data_full,models.WideResNet(num_groups=3,N=4,num_classes=NUMBER_CLASSES,k=2,start_nf=SIZE_IMAGE),metrics=accuracy)
learnFS.fit_one_cycle(150,2e-4,wd=1e-4)

print("Training supervised model with a limited set of labeled data")
learnBase = Learner(data_labeled,models.WideResNet(num_groups=3,N=4,num_classes=NUMBER_CLASSES,k=2,start_nf=SIZE_IMAGE),metrics=accuracy)
learnBase.fit_one_cycle(150,2e-4,wd=1e-4)
print("Training semi supervised model with limited set of labeled data")

"""
fit[source][test]
fit(epochs:int, lr:Union[float, Collection[float], slice]=slice(None, 0.003, None), wd:Floats=None, callbacks:Collection[Callback]=None)
Fit the model on this learner with lr learning rate, wd weight decay for epochs with callbacks.
"""
#Semi supervised trainer
learn = Learner(data_unlabeled,model,loss_func=MixupLoss(),callback_fns=[MixMatchTrainer],metrics=accuracy)
#learn.fit_one_cycle(25,2e-4,wd=0.02)

