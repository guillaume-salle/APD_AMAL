# APD Attack

We will open our code after it is accepted. For now, we provide a simple implementation here. 

# Environment
python=3.9, pytorch=1.9.1, torchvision=0.11.1, torchcam=0.3.2. 

# Prepare

## Dataset

Download from https://github.com/ylhz/tf_to_pytorch_model/tree/main/dataset and put it to `\imagenet-val\image_val_1000`. 

## Pretrained Models

Due to file size limitations, we can only provide links for pretrained models that we used to test our method. 

Download these models and put them to `\models`.

| models|
| ------------------------------------------------------------ |
| [tf2torch_inception_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_inception_v3.npy) |
| [tf2torch_inception_v4](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_inception_v4.npy) |
|[tf2torch_resnet_v2_50](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_resnet_v2_50.npy)|
|[tf2torch_resnet_v2_101](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_resnet_v2_101.npy)|
|[tf2torch_resnet_v2_152](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_resnet_v2_152.npy)|
| [tf2torch_inc_res_v2](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_inc_res_v2.npy) |
| [tf2torch_adv_inception_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_adv_inception_v3.npy) | 
| [tf2torch_ens3_adv_inc_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_ens3_adv_inc_v3.npy) 
| [tf2torch_ens4_adv_inc_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_ens4_adv_inc_v3.npy) 
| [tf2torch_ens_adv_inc_res_v2](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_ens_adv_inc_res_v2.npy) 

# Run

Just run without any parameters. 

If you want to run it without our APD attack method, just set `Dropout=False` in our codes. 


