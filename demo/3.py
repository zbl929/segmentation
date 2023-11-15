from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot

config_path = r'D:\CV_project\mmsegmentation-0.23.0\tools\logs\pretrain_nextvit_seghead_newweight_channel512\nextvit_seg_newloss.py'
checkpoint_path = r'D:\CV_project\mmsegmentation-0.23.0\tools\logs\pretrain_nextvit_seghead_newweight_channel512\iter_18000.pth'
img_path = r'D:\CV_project\mmsegmentation-0.23.0\tools\2000.jpg'

# 从配置文件和权重文件构建模型
model = init_segmentor(config_path, checkpoint_path, device='cuda:0')
# 推理给定图像
result = inference_segmentor(model, img_path)
# 展示分割结果
vis_image = show_result_pyplot(model, img_path, result)


vis_iamge = show_result_pyplot(model, img_path, result, out_file='work_dirs/result.png')
