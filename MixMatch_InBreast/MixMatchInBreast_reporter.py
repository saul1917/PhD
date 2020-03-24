from fastai.vision import *
from fastai.callbacks import CSVLogger
from numbers import Integral
import logging
#from utilities.InBreastDataset import InBreastDataset
from utilities.run_context import RunContext
import utilities.cli as cli
import seaborn as sns
#WHAT TRANSFORMATIONS ARE USED?
#IMAGE DIMENSIONS?

SIZE_IMAGE = 100
NUMBER_CLASSES = 6
class MultiTransformLabelList(LabelList):
    def __getitem__(self, idxs: Union[int, np.ndarray]) -> 'LabelList':
        """
        Create K transformed images for the unlabeled data
        :param idxs:
        :return:
        """
        "return a single (x, y) if `idxs` is an integer or a new `LabelList` object if `idxs` is a range."
        global args
        idxs = try_int(idxs)
        if isinstance(idxs, Integral):
            if self.item is None:
                x, y = self.x[idxs], self.y[idxs]
            else:
                x, y = self.item, 0
            if self.tfms or self.tfmargs:
                # I've changed this line to return a list of augmented images
                x = [x.apply_tfms(self.tfms, **self.tfmargs) for _ in range(args.K_transforms)]
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve': False})
            if y is None: y = 0
            return x, y
        else:
            return self.new(self.x[idxs], self.y[idxs])



def MixmatchCollate(batch):
    """
    # I'll also need to change the default collate function to accomodate multiple augments
    :param batch:
    :return:
    """
    batch = to_data(batch)
    if isinstance(batch[0][0], list):
        batch = [[torch.stack(s[0]), s[1]] for s in batch]
    return torch.utils.data.dataloader.default_collate(batch)





class MixupLoss(nn.Module):
    """
    Implements the mixup loss
    """
    def forward(self, preds, target, unsort=None, ramp=None, num_labeled=None):
        global args
        """
        Implements the forward pass of the loss function
        :param preds: predictions of the model
        :param target: ground truth targets
        :param unsort: ?
        :param ramp: ramp weight
        :param num_labeled:
        :return:
        """
        if unsort is None:
            return F.cross_entropy(preds,target)
        preds = preds[unsort]
        preds_l = preds[:num_labeled]
        preds_ul = preds[num_labeled:]
        preds_l = torch.log_softmax(preds_l,dim=1)
        preds_ul = torch.softmax(preds_ul,dim=1)
        loss_x = -(preds_l * target[:num_labeled]).sum(dim=1).mean()
        loss_u = F.mse_loss(preds_ul, target[num_labeled:])
        self.loss_x = loss_x.item()
        self.loss_u = loss_u.item()
        return loss_x + args.lambda_unsupervised * ramp * loss_u


class MixMatchImageList(ImageList):
    """
    Custom ImageList with filter function
    """
    def filter_train(self, num_items, seed = 2343):
        """
        Takes a number of observations as labeled, assumes that the evaluation observations are in the test folder
        :param num_items:
        :param seed: The seed is fixed for reproducibility
        :return: return the filtering function by itself
        """
        # -2 for the MNIST structure!
        train_idxs = np.array([i for i, observation in enumerate(self.items) if Path(observation).parts[-3] != "test"])
        valid_idxs = np.array([i for i, observation in enumerate(self.items) if Path(observation).parts[-3] == "test"])
        # for reproducibility
        np.random.seed(seed)
        # keep the number of items desired, 500 by default
        keep_idxs = np.random.choice(train_idxs, num_items, replace=False)

        logger.info("Number of labeled observations: " + str(len(keep_idxs)))
        logger.info("First labeled id" + str(keep_idxs[0]))
        logger.info("Number of validation observations: " + str(len(valid_idxs)))
        logger.info("Number of training observations " + str(len(train_idxs)))
        self.items = np.array([o for i, o in enumerate(self.items) if i in np.concatenate([keep_idxs, valid_idxs])])
        return self


class MixMatchTrainer(LearnerCallback):
    """
    Mix match trainer functions
    """
    _order = -20

    def on_train_begin(self, **kwargs):
        """
        Callback used when the trainer is beginning, inits variables
        :param kwargs:
        :return:
        """
        global data_labeled
        self.l_dl = iter(data_labeled.train_dl)
        #metrics recorder
        self.smoothL, self.smoothUL = SmoothenValue(0.98), SmoothenValue(0.98)
        #metrics to be displayed in the table
        self.recorder.add_metric_names(["l_loss", "ul_loss"])
        self.it = 0

    def mixup(self, a_x, a_y, b_x, b_y):
        """
        Mixup augments data by mixing labels and pseudo labels and its observations
        :param a_x:
        :param a_y:
        :param b_x:
        :param b_y:
        :param alpha:
        :return:
        """
        global args
        alpha = args.alpha_mix
        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)
        x = l * a_x + (1 - l) * b_x
        y = l * a_y + (1 - l) * b_y
        return x, y

    def sharpen(self, p):
        global args
        """
        Sharpens the distribution output, to encourage confidence
        :param p:
        :param T:
        :return:
        """
        T = args.T_sharpening
        u = p ** (1 / T)
        return u / u.sum(dim=1, keepdim=True)

    def on_batch_begin(self, train, last_input, last_target, **kwargs):
        """
        Called on batch training at the begining
        :param train:
        :param last_input:
        :param last_target:
        :param kwargs:
        :return:
        """
        global data_labeled, args
        if not train: return
        try:
            x_l, y_l = next(self.l_dl)
        except:
            self.l_dl = iter(data_labeled.train_dl)
            x_l, y_l = next(self.l_dl)

        x_ul = last_input

        with torch.no_grad():
            #calculates the pseudo sharpened labels
            ul_labels = self.sharpen(
                torch.softmax(torch.stack([self.learn.model(x_ul[:, i]) for i in range(x_ul.shape[1])], dim=1),
                              dim=2).mean(dim=1))
        #create torch array of unlabeled data
        x_ul = torch.cat([x for x in x_ul])

        #WE CAN CALCULATE HERE THE CONFIDENCE COEFFICIENT

        ul_labels = torch.cat([y.unsqueeze(0).expand(args.K_transforms, -1) for y in ul_labels])

        l_labels = torch.eye(data_labeled.c).cuda()[y_l]

        w_x = torch.cat([x_l, x_ul])
        w_y = torch.cat([l_labels, ul_labels])
        idxs = torch.randperm(w_x.shape[0])
        #create mixed input and targets
        mixed_input, mixed_target = self.mixup(w_x, w_y, w_x[idxs], w_y[idxs])
        bn_idxs = torch.randperm(mixed_input.shape[0])
        unsort = [0] * len(bn_idxs)
        for i, j in enumerate(bn_idxs): unsort[j] = i
        mixed_input = mixed_input[bn_idxs]

        ramp = self.it / 3000.0 if self.it < 3000 else 1.0
        return {"last_input": mixed_input, "last_target": (mixed_target, unsort, ramp, x_l.shape[0])}

    def on_batch_end(self, train, **kwargs):
        """
        Add the metrics at the end of the batch training
        :param train:
        :param kwargs:
        :return:
        """
        if not train: return
        self.smoothL.add_value(self.learn.loss_func.loss_x)
        self.smoothUL.add_value(self.learn.loss_func.loss_u)
        self.it += 1

    def on_epoch_end(self, last_metrics, **kwargs):
        """
        When the epoch ends, add the accmulated metric values
        :param last_metrics:
        :param kwargs:
        :return:
        """
        return add_metrics(last_metrics, [self.smoothL.smooth, self.smoothUL.smooth])


def train_mix_match():
    """
    Train the mix match model
    :param path_labeled:
    :param path_unlabeled:
    :param number_epochs:
    :param learning_rate:
    :param mode:
    :return:
    """
    global data_labeled
    global logger, context, args
    learning_rate = args.lr
    path_labeled = args.path_labeled
    path_unlabeled = args.path_unlabeled
    if(args.path_unlabeled == ""):
        path_unlabeled = path_labeled
    number_epochs = args.epochs
    logger = logging.getLogger('main')
    context = RunContext(logging, args)

    # Grab file path to cifar dataset. Will download data if not present
    #path = untar_data(URLs.CIFAR)
    #path = DEFAULT_PATH

    meanDatasetComplete = [0.1779, 0.1779, 0.1779]
    stdDatasetComplete =  [0.2539, 0.2539, 0.2539]
    inbreast_stats = (meanDatasetComplete, stdDatasetComplete)
    logger.info("Loading labeled data from: " + path_labeled)
    logger.info("Loading unlabeled data from: " + path_unlabeled)
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
    data_labeled = (MixMatchImageList.from_folder(path_labeled)
                    .filter_train(args.number_labeled)  # Use 500 labeled images for traning
                    .split_by_folder(valid="test")  # test on all 10000 images in test set
                    .label_from_folder()
                    .transform(get_transforms(max_zoom = 1, max_warp =None, p_affine = 0 ), size=SIZE_IMAGE)
                    # On windows, must set num_workers=0. Otherwise, remove the argument for a potential performance improvement
                    .databunch(bs = args.batch_size, num_workers = args.workers)
                    .normalize(inbreast_stats))


    #normalize_funcs(mean:FloatTensor, std:FloatTensor, do_x:bool=True, do_y:bool=False)
    train_set = set(data_labeled.train_ds.x.items)

    #load the unlabeled data
    src = (ImageList.from_folder(path_unlabeled)
           .filter_by_func(lambda x: x not in train_set)
           .split_by_folder(valid="test"))


    src.train._label_list = MultiTransformLabelList
    #https://docs.fast.ai/vision.transform.html
    #data not in the train_set and splitted by test folder is used as unlabeled
    data_unlabeled = (src.label_from_folder()
                      .transform(get_transforms(max_zoom=1, max_warp=None, p_affine=0), size=SIZE_IMAGE)
                      .databunch(bs = args.batch_size, collate_fn=MixmatchCollate, num_workers=10)
                      .normalize(inbreast_stats))

    # Databunch with all 50k images labeled, for baseline
    data_full = (ImageList.from_folder(path_labeled)
                 .split_by_folder(valid="test")
                 .label_from_folder()
                 .transform(get_transforms(max_zoom=1, max_warp=None, p_affine=0), size=SIZE_IMAGE)
                 .databunch(bs =  args.batch_size, num_workers = args.workers)
                 .normalize(inbreast_stats))

    #start_nf the initial number of features
    """
    Wide ResNet with num_groups and a width of k.
    Each group contains N blocks. start_nf the initial number of features. Dropout of drop_p is applied in between the two convolutions in each block. The expected input channel size is fixed at 3.
    Structure: initial convolution -> num_groups x N blocks -> final layers of regularization and pooli
    """
    if(args.model == "wide_resnet"):
        model = models.WideResNet(num_groups=3,N=4,num_classes=NUMBER_CLASSES,k = 2,start_nf=SIZE_IMAGE)
    elif(args.model == "densenet"):
        model = models.densenet121(num_classes=NUMBER_CLASSES)
    elif(args.model == "squeezenet"):
        model = models.squeezenet1_1(num_classes=NUMBER_CLASSES)

    if(args.mode == "fully_supervised"):
        logger.info("Training fully supervised model")
        #Edit: We can find the answer ‘Note that metrics are always calculated on the validation set.’ on this page: https://docs.fast.ai/training.html 42.
        learnFS = Learner(data_full,models.WideResNet(num_groups=3,N=4,num_classes=NUMBER_CLASSES,k=2,start_nf=SIZE_IMAGE, callback_fns=[CSVLogger]),metrics=[accuracy])
        learnFS.fit_one_cycle(number_epochs, learning_rate, wd=args.weight_decay)

    if(args.mode == "partial_supervised"):
        logger.info("Training supervised model with a limited set of labeled data")
        learnBase = Learner(data_labeled,models.WideResNet(num_groups=3,N=4,num_classes=NUMBER_CLASSES,k=2,start_nf=SIZE_IMAGE),metrics=[accuracy], callback_fns=[CSVLogger])
        learnBase.fit_one_cycle(number_epochs, learning_rate, wd=args.weight_decay)


    """
    fit[source][test]
    fit(epochs:int, lr:Union[float, Collection[float], slice]=slice(None, 0.003, None), wd:Floats=None, callbacks:Collection[Callback]=None)
    Fit the model on this learner with lr learning rate, wd weight decay for epochs with callbacks.
    """
    #300 epochs and stuck at 0.8 loss
    if(args.mode == "ssdl"):
        logger.info("Training semi supervised model with limited set of labeled data")
        #https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
        learn = Learner(data_unlabeled, model,loss_func = MixupLoss(),callback_fns=[MixMatchTrainer, CSVLogger],metrics=[accuracy, Precision(average = "macro"), Recall(average = "micro")])
        #learn.fit_one_cycle(600,1e-4,wd=1e-4)
        learn.fit_one_cycle(number_epochs, learning_rate, wd=args.weight_decay)

    logged_frame = learn.csv_logger.read_logged_file()
    context.write_run_log(logged_frame, args.results_file_name)

if __name__ == '__main__':
    global args
    #But no matter what I try my model never generalizes. My training loss and validation loss are diverging almost immediately, and my training loss << validation loss. So I’m overfitting.
    args = cli.parse_commandline_args()
    train_mix_match()