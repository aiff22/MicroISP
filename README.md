## MicroISP: Processing 32MP Photos on Mobile Devices with Deep Learning

<br/>

<img src="https://people.ee.ethz.ch/~ihnatova/demo_microisp/MicroISP_teaser.jpg"/>

#### 1. Overview [[Paper]](https://arxiv.org/pdf/2211.06770) [[Project Webpage]](http://people.ee.ethz.ch/~ihnatova/microisp.html)

This repository provides the implementation of the RAW-to-RGB mapping approach and MicroISP CNN presented in [this paper](https://arxiv.org/pdf/2211.06770). The model is trained to convert **RAW Bayer data** obtained directly from mobile camera sensor into photos captured with a professional medium format 102MP [Fujifilm GFX100](https://www.dpreview.com/reviews/fujifilm-gfx-100-review) camera, thus replacing the entire hand-crafted ISP camera pipeline. The provided pre-trained MicroISP model can be used to generate full-resolution **32MP photos** from RAW (DNG) image files captured using the Sony Exmor IMX586 camera sensor directly on mobile devices.

<br/>

#### 2. Prerequisites

- Python: scipy, numpy, imageio and pillow packages
- [TensorFlow 2.X](https://www.tensorflow.org/install/)
- [Optioinal] Nvidia GPU  + [CUDA cuDNN](https://developer.nvidia.com/cudnn)

<br/>

#### 3. MicroISP CNN

<br/>

<img src="https://people.ee.ethz.ch/~ihnatova/demo_microisp/microisp_architecture.png" alt="drawing" width="1000"/>

<br/>

The model accepts the raw RGBG Bayer data coming directly from the camera sensor. The input is then grouped in 4 feature maps corresponding to each of the four RGBG color channels using the space-to-depth op. Next, this input is processed in parallel in 3 model branches corresponding to the R, G and B color channels and consisting of N residual building blocks. After applying the depth-to-space op at the end of each branch, their outputs are concatenated into the reconstructed RGB photo.

The proposed MicroISP model contains only layers supported by the **Neural Networks API 1.2**, and thus can run on any NNAPI-compliant AI accelerator (such as NPU, APU, DSP or GPU) available on mobile devices with Android 10 and above. The size of the MicroISP network is only **158 KB** when exported for inference using the TFLite FP32 format. The model consumes around **90**, **475** and **975MB** of RAM when processing **FullHD**, **12MP** and **32MP** photos on mobile GPUs, respectively. Its GPU runtimes on various platforms for images of different resolutions are provided below: 

<br/>

<img src="https://people.ee.ethz.ch/~ihnatova/demo_microisp/MicroISP_Runtime.png" alt="drawing" width="1000"/>

<br/>

#### 4. Test the provided pre-trained models on full-resolution RAW image files

```bash
python inference.py
```

The model will then process DNG images from the ``sample_RAW_photos`` directory and save the resulting RGB/PNG images to the ``sample_visual_results`` folder.

<br/>

#### 5. Folder structure

>```pretrained_weights/```   &nbsp; - &nbsp; the folder with the provided pre-trained MicroISP model <br/>
>```sample_RAW_photos/```        &nbsp; - &nbsp; the folder with sample RAW/DNG images from the Fujifilm UltraISP Dataset <br/>
>```model.py```           &nbsp; - &nbsp; TensorFlow MicroISP implementation <br/>
>```inference.py```     &nbsp; - &nbsp; applying the pre-trained model to full-resolution test images <br/>
>```export_to_tflite.py```      &nbsp; - &nbsp; model export to TensorFlow Lite format for on-device deployment <br/>

<br/>

#### 6. License

Copyright (C) 2024 Andrey Ignatov. All rights reserved.

Licensed under the [CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

The code is released for academic research use only.

<br/>

#### 7. Citation

```
@inproceedings{ignatov2022microisp,
  title={MicroISP: Processing 32MP Photos on Mobile Devices with Deep Learning},
  author={Ignatov, Andrey and Sycheva, Anastasia and Timofte, Radu and Tseng, Yu and Xu, Yu-Syuan and Yu, Po-Hsiang and Chiang, Cheng-Ming and Kuo, Hsien-Kai and Chen, Min-Hung and Cheng, Chia-Ming and others},
  booktitle={European Conference on Computer Vision},
  pages={729--746},
  year={2022},
  organization={Springer}
}
```
<br/>

#### 8. Any further questions?

```
Please contact Andrey Ignatov (andrey@vision.ee.ethz.ch) for more information
```
