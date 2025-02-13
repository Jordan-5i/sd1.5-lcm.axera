# 模型转换

在 PC 上完成 ONNX 模型导出和 axmodel 模型编译。

## 安装依赖

```
git clone https://github.com/BUG1989/sd1.5-lcm.axera.git
cd model_convert
pip install -r requirements.txt
```

## 导出模型（Huggingface -> ONNX）

下载 Huggingface 上对应的 Repo
```
huggingface-cli download --resume-download latent-consistency/lcm-lora-sdv1-5 --local-dir latent-consistency/lcm-lora-sdv1-5
huggingface-cli download --resume-download Lykon/dreamshaper-7 --local-dir Lykon/dreamshaper-7
```

运行脚本 `sd15_export_onnx.py` 导出 unet 和 vae 的 onnx 模型
```
python sd15_export_onnx.py --input_path Lykon/dreamshaper-7/ --input_lora_path latent-consistency/lcm-lora-sdv1-5/ --output_path output_onnx/
```

完成后如下所示
```
qtang@gpux2:~/sd15-export$ tree -L 1 output_onnx/
output_onnx/
├── cc6a243a-b7a8-11ef-bb2a-9d527016cd35
├── sd15_vae_decoder.onnx
├── sd15_vae_decoder_sim.onnx
├── time_input.npy
├── unet
└── unet_sim_cut.onnx
```

## 生成量化数据集

运行脚本 `sd15_lora_prepare_data.py` 准备 Pulsar2 编译依赖的 Calibration 数据集
```
python sd15_lora_prepare_data.py
```

完成后如下所示
```
qtang@gpux2:~/sd15-export$ tree -L 1 calib_data_unet
calib_data_unet
├── data_0.npy
......
├── data_9.npy
└── data.tar
qtang@gpux2:~/sd15-export$ tree -L 1 calib_data_vae
calib_data_vae
├── data_0_0.npy
......
├── data_9_3.npy
└── data.tar
```

## 模型转换

**unet**
```
pulsar2 build --input output_onnx/unet_sim_cut.onnx --config unet_u16.json --output_dir output_unet --output_name unet.axmodel
```

**vae**
```
pulsar2 build --input output_onnx/sd15_vae_decoder_sim.onnx --config vae_u16.json --output_dir output_vae --output_name vae.axmodel
```

## Inpainting 模型转换

## 导出模型（Huggingface -> ONNX）

```
huggingface-cli download --resume-download runwayml/stable-diffusion-inpainting --local-dir runwayml/stable-diffusion-inpainting
```

运行脚本 `sd15_inpaint_export_onnx.py` 导出 unet / vae encoder / vae decoder 的 onnx 模型
```
python sd15_inpaint_export_onnx.py --input_path runwayml/stable-diffusion-inpainting --input_lora_path latent-consistency/lcm-lora-sdv1-5/ --output_path output_onnx_inpaint/
```
验证导出的ONNX是否正确，运行`run_inpaint_onnx.py`，会保存一张png图片，用来验证ONNX是否正确
```
python run_inpaint_onnx.py
```

## 生成量化数据集, Calibration 数据集在inpaint_samples中
```
python sd15_inpaint_lora_prepare_data.py
```

## 模型转换
**unet**
```
pulsar2 build --input output_onnx_inpaint/unet_sim_cut.onnx --config unet_u16.json --output_dir output_unet_inpaint --output_name unet.axmodel
```

**vae encoder**
```
# 修改 vae_u16.json 中的 calibration_dataset 指向的路径
pulsar2 build --input output_onnx_inpaint/sd15_vae_encoder_sim.onnx --config vae_u16.json --output_dir output_vae_encoder_inpaint --output_name vae_encoder.axmodel
```

**vae decoder**
```
pulsar2 build --input output_onnx_inpaint/sd15_vae_decoder_sim.onnx --config vae_u16.json --output_dir output_vae_decoder_inpaint --output_name vae_decoder.axmodel
```

## ControlNet 模型转换
```
huggingface-cli download --resume-download lllyasviel/sd-controlnet-canny --local-dir lllyasviel/sd-controlnet-canny
huggingface-cli download --resume-download runwayml/stable-diffusion-v1-5 --local-dir runwayml/stable-diffusion-v1-5
```

运行脚本 `sd15_controlnet_export_onnx.py` 导出 unet / vae encoder / vae decoder / controlnet 的 onnx 模型
```
python sd15_controlnet_export_onnx.py --input_path runwayml/stable-diffusion-v1-5 --input_lora_path latent-consistency/lcm-lora-sdv1-5/ --input_controlnet_path lllyasviel/sd-controlnet-canny --output_path output_onnx_controlnet
```
验证导出的onnx是否正确，运行`run_controlnet_onnx.py`，会保存一张png图片，用来验证ONNX是否正确.

把 unet 和 controlnet 提前计算好的 time_embedding 的npy文件copy 到models/controlnet/
```
python run_controlnet_onnx.py 
```