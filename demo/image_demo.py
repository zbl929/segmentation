# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default=r"../tools/img_1.png",help='Image file')
    parser.add_argument('--config',default=r'/home/zhangbulin/mmseg-0.23.0/tools/logs_zbldata_2class/nextvit_small_lighthead_new/nextvit_lighthead_new.py', help='Config file')
    parser.add_argument('--checkpoint',default=r'/home/zhangbulin/mmseg-0.23.0/tools/logs_zbldata_2class/nextvit_small_lighthead_new/best_mIoU_iter_9000.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:8', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='zblWaterDataset_2class',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        get_palette(args.palette),
        opacity=args.opacity)


if __name__ == '__main__':
    main()
