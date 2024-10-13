# Prepare the Dataset and Checkpoints

## Download Dataset

Download the dataset from the  [Google Drive](https://drive.google.com/drive/folders/11H9Fo92Nv46-dAXLazO4Igg_iWqjk9QK) and place them in `Dataset` folder in the following structure:

```cmd
<Dataset>
|-- <00_List>
    |-- train_val.txt
    |-- test.txt
|-- <00_Visible_Image>
    |-- LAB00162.png
    |-- LAB00165.png
    ...
|-- <02_Infrared_Image>
    |-- LAB00162.png
    |-- LAB00165.png
    ...
|-- <03_Fusion_Image>
    |-- LAB00162.png
    |-- LAB00165.png
    ...
|-- <04_Ground_Truth>
    |-- LAB00162.png
    |-- LAB00165.png
    ...
```

## Download Checkpoints

Download the pretrained checkpoints form [Google Drive](https://drive.google.com/drive/folders/1XFzpnGVtTY0T5EkKUuiS6tetedPyCx17) and place them in the following structure:

```cmd
<IRFusionFormer>
|-- <MCNet>
    |-- <model>
    	|-- <ResNet>
    		|-- resnet101-5d3b4d8f.pth
    |-- <log>
    	|-- <MCNet>
    		|-- <snapshot>
    			|-- MCNet.pth
|-- <DeepCrack>
	|-- <checkpoints>
		|-- DeepCrack_CT260_FT1.pth
		|-- DeepCrack.pth
|-- <CRM>
    |-- <pretrained>
		|-- swin_base_patch4_window7_224.pkl
		...
	|-- <pretrained>
		|-- CRM_swin_B.ckpt
		...
|-- <DELIVER>
    |-- <checkpoints>
    	|-- <pretrained>
			|-- <segformer>
				|-- mit_b4.pth
	|-- <output>
		|-- CMNeXt_CMNeXt-B4.pth
|-- <IRFusionFormer>
    |-- <pretrained>
		|-- <segformer-b3-finetuned-ade-512-512>
			|-- tf_model.h5
			...
	|-- <ckpt>
		|-- <IRFusionFormer>
			|-- IRFusionFormer.ckpt
		|-- <U-Net>
			|-- U-Net.ckpt
		|-- <UNet++>
			|-- NestedUNet.ckpt
		|-- <DeepLabV3+>
			|-- DeepLabV3Plus.ckpt
		|-- <CrackFormer>
			|-- CrackFormer.ckpt

```