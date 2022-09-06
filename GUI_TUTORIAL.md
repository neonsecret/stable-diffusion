# Stable Diffusion GUI tutorial (for now Windows only)
### Step 0. Download the packed GUI
If you want to go the simple way, 
[download this](https://mega.nz/file/TEw3HCxA#SbDyov9ZXIGH8FEJ1OeXyOU7D_G4ykqg8WkZdvIr1nM) (8 gb).<br>

If you already have a model.ckpt file,
[download this](https://mega.nz/file/aFZTGLJA#-jpo2uuZu0KvsUeS6cpPTUcwLpolowszY3Vcch5IbUA) (4 gb).
### Step 1. 
Extract the downloaded archive.

If you downloaded the 4 gb archive, put the model.ckpt file to StableDiffusionGui\_internal\stable_diffusion\models\ldm\stable-diffusion-v1

<img src="assets/tutorial_imgs/step 1.jpg"/>


also install git if not installed
https://git-scm.com/downloads

### Step 2. 
Your folder contents should look like this
<img src="assets/tutorial_imgs/step 2.jpg"/>

### Step 3. 
run the 
```
1) install.bat
```
file. It should produce the output as in the screenshot.
<img src="assets/tutorial_imgs/step 3.jpg"/>

### Step 4.1
Each file explained:

``` SD_OPT) run optimized txt2img.bat ``` is used to run the txt2img gradio interface. after double-clicking it, go to step 5.

```SD_OPT) run optimized img2img.bat``` is used to run the img2img gradio interface. after double-clicking it, go to step 5.

```SD_OPT) run optimized img2img inpainting.bat``` is used to run the img2img inpainting gradio interface. after double-clicking it, go to step 5.

<br> Choose only one of these three if you are limited by resources or want to generate bigger-resolution images. Otherwise, go to step 4.2.

<img src="assets/tutorial_imgs/step 4.1.jpg"/>

### Step 4.2
Obviously the same as 4.1 but better speed, though more resource consumption.
<img src="assets/tutorial_imgs/step 4.2.jpg"/>

### Step 4.3
If you encounter this error, just press 'OK', it doesn't mean anything.<br>
<img src="assets/tutorial_imgs/step 5.jpg"/>

### Step 5
After clicking one of the 6 .bat files, you should see output like this.
<br>
Choose the red link in 99% cases.
Choose the blue one if you have troubles with the red one. They  lead to the same page.
Copy your link of choice and paste it into your web-browser. (Chrome Firefox whatever)
<img src="assets/tutorial_imgs/step 6.png"/>

### Step +. Gradio interface explained.
Once you open the page, you should see interface like this. It may vary because each mode has it's interface. 

Though I think it's intuitive, I will explain some of the params:
- ddm_steps: usally 50 is the fine value, lower it if you want faster results
- n_iter: number of generated images, will appear in a grid.
- scale: how much your prompt will influence the image, experiment with it
- turbo: if you have memory errors, try disabling it.
- sampler: different sampler may produce a bit different results.
<br>
Other params are better to be left as-is.
<img src="assets/tutorial_imgs/step 7.jpg"/>
