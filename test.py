import torch
import time
import argparse
from model import fusion_refine #, Discriminator
# from train_dataset import dehaze_train_dataset
# from test_dataset import dehaze_test_dataset
from val_dataset import dehaze_val_dataset, dehaze_val_dataset_ohaze
from torch.utils.data import DataLoader
import os
# from torchvision.models import vgg16
# from utils_test import to_psnr, to_ssim_skimage, test_generate, image_stick, image_stick_ohaze
from utils_test import test_generate, image_stick, image_stick_ohaze
# from tensorboardX import SummaryWriter
# import torch.nn.functional as F
# from perceptual import LossNetwork
from torchvision.utils import save_image as imwrite
# from pytorch_msssim import msssim

from config import get_config
from models import build_model
from collections import OrderedDict ########## ---- CH code ---- ##########
from tqdm import tqdm ########## ---- CH code ---- ##########

# # --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='RCAN-Dehaze-teacher')
# parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
# parser.add_argument('-train_batch_size', help='Set the training batch size', default=20, type=int)
# parser.add_argument('-train_epoch', help='Set the training epoch', default=10000, type=int)
# parser.add_argument('--train_dataset', type=str, default='')
parser.add_argument('--data_dir', type=str, default='./input_data')
parser.add_argument('--model_save_dir', type=str, default='./output_result')
# parser.add_argument('--log_dir', type=str, default=None)

# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('--predict_result', type=str, default='./output_result')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1,  type=int)
parser.add_argument('--vgg_model', default='', type=str, help='load trained model or not')
parser.add_argument('--imagenet_model', default='', type=str, help='load trained model or not')
parser.add_argument('--rcan_model', default='', type=str, help='load trained model or not')
# parser.add_argument('--ckpt_path', default='./weights/best.pkl', type=str, help='path to model to be loaded')
parser.add_argument('--hazy_data', default='', type=str, help='apply on test data or val data')
parser.add_argument('--cropping', default='4', type=int, help='crop the 4k*6k image to # of patches for testing') # use 4 for >40GB memory, else use 6
parser.add_argument('--datasets', nargs='+', default=['Test'])  ## ch add
parser.add_argument('--weights', nargs='+', default=['dehaze.pkl'], help='List of weight file names as strings') ## ch add

# --- SwinTransformer Parameter --- #
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )         # required
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
# easy config modification
parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
parser.add_argument('--data-path', type=str, help='path to dataset')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

# distributed training
#parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel') # required

# for acceleration
parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')           # --fused_window_process
parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

########## ---- Start of CH code: Set Variables ---- ##########
#### -------------------------- SET VARIABLES ----------------------------------------------
# list_weight_files = ["./weights/best.pkl"]   ## original weights
# list_weight_files = ["2025-04-02_10-41-25_NHNH2RB10_epoch01000.pkl", "2025-04-02_10-41-25_NHNH2RB10_epoch02000.pkl", "2025-04-02_10-41-25_NHNH2RB10_epoch03000.pkl", "2025-04-02_10-41-25_NHNH2RB10_epoch04000.pkl", "2025-04-02_10-41-25_NHNH2RB10_epoch05000.pkl", "2025-04-02_10-41-25_NHNH2RB10_epoch06000.pkl", "2025-04-02_10-41-25_NHNH2RB10_epoch07000.pkl", "2025-04-02_10-41-25_NHNH2RB10_epoch08000.pkl", "2025-04-03_13-39-05_NHNH2_epoch01000.pkl", "2025-04-03_13-39-05_NHNH2_epoch02000.pkl", "2025-04-03_13-39-05_NHNH2_epoch03000.pkl", "2025-04-03_13-39-05_NHNH2_epoch04000.pkl", "2025-04-03_13-39-05_NHNH2_epoch05000.pkl", "2025-04-03_13-39-05_NHNH2_epoch06000.pkl", "2025-04-03_13-39-05_NHNH2_epoch07000.pkl", "2025-04-03_13-39-05_NHNH2_epoch08000.pkl"]  ## SET VARIABLE
# list_weight_files = ["2025-04-03_13-39-05_NHNH2_epoch08000.pkl"]  ## SET VARIABLE

# # test_datasets = ["Dense", "DNH_1600x1200", "SMOKE_1600x1200_test"]   ## SET VARIABLE
# test_datasets = ["Dense"]   ## SET VARIABLE

multiple_gpus = True  ## SET VARIABLE (Code by Caitlin)
#### -------------------------- SET VARIABLES ----------------------------------------------
########## ---- End of CH code ---- ##########

## ch add
def normalize_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        while k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v
    return new_state_dict

## TEMP
def compare_state_dicts(model, checkpoint_path, output_file='state_dict_comparison.txt', device='cpu'):
    with open(output_file, 'w') as f:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            f.write("Loaded 'state_dict' from checkpoint\n")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            f.write("Loaded 'model_state_dict' from checkpoint\n")
        else:
            state_dict = checkpoint
            f.write("Checkpoint loaded directly as state_dict\n")

        model_state = model.state_dict()

        model_keys = set(model_state.keys())
        checkpoint_keys = set(state_dict.keys())

        missing_in_checkpoint = model_keys - checkpoint_keys
        extra_in_checkpoint = checkpoint_keys - model_keys

        # Print expected model keys
        f.write("\nExpected model state_dict keys:\n")
        for k in sorted(model_keys):
            f.write(f"{k}\n")

        # Print checkpoint keys
        f.write("\nCheckpoint keys:\n")
        for k in sorted(checkpoint_keys):
            f.write(f"{k}\n")

        # Print mismatches
        f.write("\nMissing keys in checkpoint (needed by model but not found):\n")
        for k in sorted(missing_in_checkpoint):
            f.write(f"{k}\n")

        f.write("\nExtra keys in checkpoint (not used by model):\n")
        for k in sorted(extra_in_checkpoint):
            f.write(f"{k}\n")

    print(f"[INFO] Comparison written to {output_file}")

args = parser.parse_args()

for dataset in args.datasets:
    print(f"-- Testing on dataset {dataset}: --")
    # test_data_dir = os.path.join(args.test_dir, dataset)  ## ch add  
    list_weight_files = args.weights ## ch add

    # test_dataset = dehaze_test_dataset(test_data_dir)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=2)
    tested_on = dataset ## ch add

    for weight_file in list_weight_files:
        # --- output picture and check point --- #
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        output_dir = os.path.join(args.model_save_dir,'')

        weight_name = os.path.splitext(os.path.basename(weight_file))[0]
        final_output_dir = os.path.join(output_dir, f"{tested_on}_{weight_name}")  # save each set of images in a new directory with the weight name

        # weight_file = os.path.join("./weights", weight_file)  ## weights must be saved in weights directory

        val_dataset = os.path.join(args.data_dir, dataset)        
        predict_result = args.predict_result
        test_batch_size = args.test_batch_size

        

        # --- Gpu device --- #
        device_ids = [Id for Id in range(torch.cuda.device_count())]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # --- Define the network --- #
        # --- Define the Swin Transformer V2 model --- #
        config = get_config(args)
        swv2_model = build_model(config)

        # --- Define the network --- #
        if args.imagenet_model == 'SwinTransformerV2':
            MyEnsembleNet = fusion_refine(swv2_model, args.rcan_model)
        elif args.imagenet_model == 'Res2Net':
            MyEnsembleNet = fusion_refine(args.imagenet_model, args.rcan_model)

        #SMyEnsembleNet = fusion_refine(args.imagenet_model, args.rcan_model)
        print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))

        if dataset == "OHAZE_test":
            val_dataset = dehaze_val_dataset_ohaze(val_dataset, crop_method=args.cropping)
        else:
            val_dataset = dehaze_val_dataset(val_dataset, crop_method=args.cropping)

        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        ########## ---- Start of CH code: Run tests for multiples weights ---- ##########


        weight_name = os.path.splitext(os.path.basename(weight_file))[0]
        final_output_dir = os.path.join(output_dir, f"{tested_on}_{weight_name}")  # save each set of images in a new directory with the weight name

        weight_file = os.path.join("./weights", weight_file)
        ########## ---- End of CH code ---- ##########

        # print(f"------- starting ---------")
        print(f"Testing {weight_file}:")
        comparison_log_path = os.path.join(output_dir, f'{weight_name}_state_dict_comparison.txt')
        compare_state_dicts(MyEnsembleNet, weight_file, output_file=comparison_log_path)

        if multiple_gpus: 
            # --- Multi-GPU --- #
            MyEnsembleNet = MyEnsembleNet.to(device)
            
            ########## ---- Start of CH code: To use given pkl file with parallel GPUs ---- ##########
            # checkpoint = torch.load(args.ckpt_path, map_location=device)  # Ensure checkpoint is loaded on correct device
            checkpoint = torch.load(weight_file, map_location=device)  # Ensure checkpoint is loaded on correct device

            if "model_state_dict" in checkpoint:
                print(f"found model state dict")
                state_dict = checkpoint["model_state_dict"]  # Extract the actual state dict
            elif "state_dict" in checkpoint:
                print(f"found state dict")
                state_dict = checkpoint["state_dict"]  # Extract the actual state dict
            else:
                state_dict = checkpoint  # Direct state_dict case
                print(f"no model state dict")
            
            new_state_dict = normalize_state_dict(state_dict)
            MyEnsembleNet.load_state_dict(new_state_dict)


            # ########## ---- End of CH code ---- ##########

            # Now wrap in DataParallel
            MyEnsembleNet = torch.nn.DataParallel(MyEnsembleNet, device_ids=device_ids)
            MyEnsembleNet = MyEnsembleNet.to(device)
            # --- Load the network weight --- #
            # MyEnsembleNet.load_state_dict(torch.load(args.ckpt_path))              
           
        # --- Strat testing --- #
        with torch.no_grad():
            img_list = []
            time_list = []

            imsave_dir = output_dir
            if not os.path.exists(imsave_dir):
                os.makedirs(imsave_dir)
                print("Created output directory")
                
            MyEnsembleNet.eval()
            for batch_idx, (hazy, vertical, name) in enumerate(tqdm(val_loader)):   
            #for batch_idx, (hazy, vertical, hazy_1, hazy_2, hazy_3,hazy_4, hazy_5, hazy_6) in enumerate(val_loader):
                # print(len(val_loader))

                if not os.path.exists(final_output_dir + '/'): ## ch add
                    os.makedirs(final_output_dir + '/') ## ch add

                # for OHAZE
                if dataset == "OHAZE_test":
                    index = vertical[0].to('cpu')
                    ori_shape = vertical[1]
                    img_list = []
                    
                    for input in hazy:
                        input = input.to(device)
                        img_tensor = MyEnsembleNet(input)
                        # img_list.append(img_tensor.cpu())
                        img_list.append(img_tensor) # ch add

                    final_image = image_stick_ohaze(img_list, index, ori_shape)
                    assert final_image.shape[-1] == ori_shape[-1]
                    # final_image.shape  ori_shape

                    continue

                # Below for NTIRE2023
                if args.cropping == 4:
                    img_list = []
                    
                    for input in hazy:
                        input = input.to(device)
                        #print(input.size())
                        start = time.time()
                        img_tensor = MyEnsembleNet(input)
                        end = time.time()
                        time_list.append((end - start))
                        img_list.append(img_tensor.cpu())
                
                    final_image = image_stick(img_list, vertical)
                    one_t = torch.ones_like(final_image[:,0,:,:])
                    one_t = one_t[:, None, :, :]
                    img_t = torch.cat((final_image, one_t) , 1)
                    
                    ######## Code by Caitlin #################
                    # name = os.listdir(os.path.join(args.data_dir, args.hazy_data))[batch_idx]
                    # imwrite(img_t, os.path.join(output_dir, f"{name}.png"), range=(0, 1))
                    imwrite(img_list[batch_idx], os.path.join(final_output_dir, str(name[0])))
                    ##########################################

                else:
                    # start = time.time()

                    img_tensor = test_generate(hazy, vertical, args.cropping, MyEnsembleNet, device)
                
                    # end = time.time()
                    # time_list.append((end - start))
                    img_list.append(img_tensor)


                    ######## Code by Caitlin #################
                    # print(f"find name: {val_dataset[batch_idx]}")
                    # name = str(batch_idx + 41)  # os.listdir(os.path.join(args.data_dir, args.hazy_data))[batch_idx]
                    imwrite(img_list[batch_idx], os.path.join(final_output_dir, str(name[0])))

                    ##########################################

            # time_cost = float(sum(time_list) / len(time_list))
            # print('running time per image: ', time_cost)
            #                  
        # writer.close()
