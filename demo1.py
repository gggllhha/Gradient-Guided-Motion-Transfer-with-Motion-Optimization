import os
import sys
import warnings

import yaml
from argparse import ArgumentParser
from tqdm.auto import tqdm

import imageio
import numpy as np
import torch.nn.functional as F
from skimage.transform import resize
# from skimage import img_as_ubyte

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch


from skimage.draw import ellipse as circle
from logger import Logger
import matplotlib.pyplot as plt
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork
from modules.bg_motion_predictor import BGMotionPredictor

warnings.filterwarnings("ignore")
# import ffmpeg
from os.path import splitext
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def draw_image_with_kp(image, kp_array):
    image = np.copy(image)
    spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
    kp_array = spatial_size * (kp_array + 1) / 2
    num_kp = kp_array.shape[0]
    for kp_ind, kp in enumerate(kp_array):
        rr, cc = circle(kp[1], kp[0], 5, 5, shape=image.shape[:2])
        image[rr, cc] = np.array(plt.get_cmap('gist_rainbow')(kp_ind / num_kp))[:3]
    return image


def create_image_column_with_kp(images, kp):
    image_array = np.array([draw_image_with_kp(v, k) for v, k in zip(images, kp)])
    return create_image_column(image_array)


def create_image_column(images):
    return np.concatenate(list(images), axis=0)


def create_image_grid(*args):
    out = []
    for arg in args:
        if type(arg) == tuple:
            out.append(create_image_column_with_kp(arg[0], arg[1]))
        else:
            out.append(create_image_column(arg))
    return np.concatenate(out, axis=1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='/disk/disk0/wqw/Thin-Plate-Spline-Motion-Model-main/config/taichi-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='/disk/disk0/wqw/TPS-GGMM/log_vox-256/00000099-checkpoint.pth.tar',
                        help="path to checkpoint to restore")
    parser.add_argument("--source_image", default='/disk/disk0/wqw/TPSMM/test/0000351.png', help="path to source image")
    parser.add_argument("--driving_image", default='/disk/disk0/wqw/TPSMM/test/0000121.png')
    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None, help="Set frame to start from.")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.add_argument("--audio", dest="audio", action="store_true",
                        help="copy audio to output from the driving video")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)
    parser.set_defaults(audio_on=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.safe_load(f)
    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    if torch.cuda.is_available():
        inpainting.to(opt.device_ids[0])

    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])

    if torch.cuda.is_available():
        dense_motion_network.to(opt.device_ids[0])
        kp_detector.to(opt.device_ids[0])

    if (config['model_params']['common_params']['bg']):
        bg_predictor = BGMotionPredictor()
        if torch.cuda.is_available():
            bg_predictor.to(opt.device_ids[0])

    Logger.load_cpk(opt.checkpoint, inpainting_network=inpainting, kp_detector=kp_detector,
                    bg_predictor=bg_predictor, dense_motion_network=dense_motion_network)
    # generator, kp_detector, em_decoder = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint,
    #                                                       cpu=False)
    # if torch.cuda.is_available():
    #     generator = DataParallelWithCallback(generator)
    #     kp_detector = DataParallelWithCallback(kp_detector)


    inpainting.eval()
    dense_motion_network.eval()
    kp_detector.eval()
    if bg_predictor is not None:
        bg_predictor.eval()

    source_image = imageio.imread(opt.source_image)
    driving_image = imageio.imread(opt.driving_image)
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) / 255
    driving = torch.tensor(np.array(driving_image)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) / 255
    source = source.cuda()
    driving = driving.cuda()
    kp_source = kp_detector(source)
    kp_driving = kp_detector(driving)
    bg_params = None
    if bg_predictor:
        bg_params = bg_predictor(source, driving)
    dense_motion = dense_motion_network(source_image=source, kp_driving=kp_driving,
                                        kp_source=kp_source, bg_param=bg_params,
                                        dropout_flag=False)
    out = inpainting(source, dense_motion)
    # out['kp_source'] = kp_source
    # out['kp_driving'] = kp_driving
    # colormap = plt.get_cmap('gist_rainbow')
    images = []
    # if 'occlusion_map' in out:
    #     for i in range(len(out['occlusion_map'])):
    #         occlusion_map = out['occlusion_map'][i].data.cpu().repeat(1, 3, 1, 1)
    #         occlusion_map = F.interpolate(occlusion_map, size=256).numpy()
    #         occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
    #         print(occlusion_map.shape)
    #         images.append(occlusion_map)
    # source = np.transpose(source.data.cpu().numpy(), [0, 2, 3, 1])
    # driving = np.transpose(driving.data.cpu().numpy(), [0, 2, 3, 1])
    # kp_source = kp_source['value'].data.cpu().numpy()
    # kp_driving = kp_driving['value'].data.cpu().numpy()
    prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])
    # images.append((source, kp_source))
    # images.append((driving, kp_driving))
    images.append(prediction)
    # images.append(img_with_edge)
    image = create_image_grid(*images)
    image = (255 * image).astype(np.uint8)
    imageio.imsave('/disk/disk0/wqw/TPSMM/test/output_ours.png', image)
