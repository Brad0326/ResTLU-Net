# ResTLU-Net

Paper：[A Macaca Brain extraction Model Based on U-Net Combined with Residual Structure](https://www.mdpi.com/2076-3425/12/2/260#metrics)

**Usage：**		

```bash
python Skull-removal.py -in [Your nii/nii.gz image file] -out [mask output PATH] -model [Your Best model file]
```

**Guide-Line：**

​		If your splitting effect is not very good, it means that the capture device you are using is quite different from ours. In this case, you can fine-tune the model by using `ResTLU_Net_train.py` to fine-tune the manually corrected Mask file, only need to provide 2 corrected Mask files, and train 5 epochs. 

​		***After that, you can call the new model file to separate the skull.***

** **Since we are using the free Github, it is not possible to upload larger models. If you need our latest trained model, please get it from [Baidu Netdisk](https://pan.baidu.com/s/1axLM95zTI4YFUQnV40FCwQ?pwd=9hb7).**



