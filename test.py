import os

import megengine as mge
import megengine.functional as F
import argparse
import numpy as np
import cv2

from nets import Model
from nets.utils.utils import InputPadder
from PIL import Image


def load_model(model_path):
    print("Loading model:", os.path.abspath(model_path))
    pretrained_dict = mge.load(model_path)
    model = Model(max_disp=256, mixed_precision=False, test_mode=True)

    model.load_state_dict(pretrained_dict["state_dict"], strict=True)

    model.eval()
    return model


def inference(left, right, model, n_iter=20):
    print("Model Forwarding...")
	# 1인 차원 삭제하고 넘파이로 바꾸장
    imgL=F.squeeze(left).numpy()
    imgR=F.squeeze(right).numpy()
    
    # imgL = left.transpose(2, 0, 1)
    # imgR = right.transpose(2, 0, 1)
    
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])

    imgL = mge.tensor(imgL).astype("float32")
    imgR = mge.tensor(imgR).astype("float32")

    imgL_dw2 = F.nn.interpolate(
        imgL,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    imgR_dw2 = F.nn.interpolate(
        imgR,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

    pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
    pred_disp = F.squeeze(pred_flow[:, 0, :, :]).numpy()

    return pred_disp


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8) # (2000, 1500, 3)
    #img = torch.from_numpy(img).permute(2, 0, 1).float() # torch.Size([3, 1512, 2016])
    
    img = img.transpose(2, 0, 1)# (3, 2000, 1500), <class 'numpy.ndarray'>
    img = mge.Tensor(img)# <class 'megengine.tensor.Tensor'>
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A demo to run CREStereo.")
    parser.add_argument(
        "--model_path",
        default="crestereo_eth3d.mge",
        help="The path of pre-trained MegEngine model.",
    )
    parser.add_argument(
        "--left", default="img/test/left.png", help="The path of left image."
    )
    parser.add_argument(
        "--right", default="img/test/right.png", help="The path of right image."
    )
    # parser.add_argument(
    #     "--size",
    #     default="1024x1536",
    #     help="The image size for inference. Te default setting is 1024x1536. \
    #                     To evaluate on ETH3D Benchmark, use 768x1024 instead.",
    # )
    parser.add_argument(
        "--output", default="disparity.png", help="The path of output disparity."
    )
    args = parser.parse_args()

    assert os.path.exists(args.model_path), "The model path do not exist."
    assert os.path.exists(args.left), "The left image path do not exist."
    assert os.path.exists(args.right), "The right image path do not exist."

    model_func = load_model(args.model_path)
    # left = cv2.imread(args.left)
    # right = cv2.imread(args.right)
    left = load_image(args.left)
    right = load_image(args.right)

## left right ndim4로 늘리기
    # print(type(left))# <class 'megengine.tensor.Tensor'>
    left=F.expand_dims(left,axis=0) # (1, 3, 2000, 1500)
    right=F.expand_dims(right,axis=0)
    assert left.shape == right.shape, "The input images have inconsistent shapes."

    # in_h, in_w = left.shape[:2]
    in_h, in_w = left.shape[-2:]
    
    padder = InputPadder(left.shape, divis_by=32)
    print(left.shape) # (1, 3, 2000, 1500)
    # print(left.ndim) # 4
    left_img, right_img = padder.pad(left, right)
    print(left_img.shape) # (1, 3, 2016, 1504)
    
    # ## 사이즈를 넣으면 리사이즈해줌
    # print("Images resized:", args.size)
    # eval_h, eval_w = [int(e) for e in args.size.split("x")]
    # left_img = cv2.resize(left, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    # right_img = cv2.resize(right, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

    disp = inference(left_img, right_img, model_func, n_iter=20)

    # t = float(in_w) / float(eval_w)
    # disp = cv2.resize(disp, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
    
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
    disp_vis=cv2.resize(disp_vis,dsize=(in_w,in_h),interpolation=cv2.INTER_AREA)

    parent_path = os.path.abspath(os.path.join(args.output, os.pardir))
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    cv2.imwrite('out_pad2.png', disp_vis)

    # print(type(disp)) # <class 'numpy.ndarray'>
    # print(disp.ndim) # 2
    # print(disp.shape) # (2016, 1504)
    
    #   imgR=F.squeeze(right).numpy()
    # right=F.expand_dims(right,axis=0)
    
    disp = mge.Tensor(disp)
    disp=F.expand_dims(disp, axis=0)
    disp=F.expand_dims(disp, axis=0)#expand_dims(disp, axis=0)
    # print(disp.ndim) # 2
    # print(disp.shape) # (2016, 1504)
    disp=padder.unpad(disp).numpy()
    disp.tofile('out_pad2.raw')
    print('fin')