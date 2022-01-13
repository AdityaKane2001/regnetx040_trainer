import ml_collections

#--------------- Basic config -------------------#

basics = ml_collections.ConfigDict()

basics.name = ""          # [str] Name of the model.
basics.num_classes =1000   # [int] Number of classes to be used in the model.
basics.description = ""   # [str] Description of the model.

#--------------- Preprocessing config -------------------#

preproc = ml_collections.ConfigDict()

## basic
preproc.batch_size = 1024  
preproc.image_size = 224   


## Augmentations
preproc.no_aug = False

preproc.mixup = ml_collections.ConfigDict()
preproc.cutmix = ml_collections.ConfigDict()
preproc.random_erasing = ml_collections.ConfigDict()
preproc.randaugment = ml_collections.ConfigDict()

# augmentation flags
preproc.mixup.apply = True
preproc.cutmix.apply = False
preproc.random_erasing.apply = True
preproc.randaugment.apply = True


# augmentation values
preproc.mixup.alpha = 0.2  
preproc.cutmix.alpha = 0.2
preproc.random_erasing.probability = 0.5
preproc.random_erasing.aspect_ratio = 0.3
preproc.random_erasing.min_erase = 0.02
preproc.random_erasing.max_erase = 0.4
preproc.random_erasing.fill_type = "zeros" # can be "zeros" "ones" "random"(uniform sampling)
preproc.randaugment.num_layers = 2
preproc.randaugment.magnitude = 5

#--------------- Training config -------------------#

training = ml_collections.ConfigDict()

training.ema = False
training.ema_factor = 5e-5

# optimizer values

training.optimizer = ml_collections.ConfigDict()

training.optimizer.optimizer = "adamw"
training.optimizer.base_lr = 0.001 * 8
training.optimizer.weight_decay = 5e-5
training.optimizer.momentum = 0.9
training.optimizer.label_smoothing = 0.0 

# schedule values

training.schedule = ml_collections.ConfigDict()

training.schedule.base_lr = 0.001 * 8
training.schedule.total_epochs = 300
training.schedule.warmup_epochs = 5
training.schedule.warmup_factor = 0.1
training.schedule.warmup_type = "gradual" # Can be "gradual", "step", "constant" 
training.schedule.lr_schedule = "half_cos" # Can be "half_cos", "constant"

#logging values

training.logging = ml_collections.ConfigDict()

training.logging.uses_gcs = True
training.logging.model_dir = ""
training.logging.log_dir = ""


#--------------- Main config getter -------------------#

cfg = ml_collections.ConfigDict()
cfg.basics = basics
cfg.preproc = preproc
cfg.training = training

def get_config():
    return cfg
