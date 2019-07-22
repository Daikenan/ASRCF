# ASRCF
- Visual Tracking via Adaptive Spatially-Regularized Correlation Filters

<div align="center">
  <img src="https://github.com/Daikenan/ASRCF/blob/master/faceocc1.gif" width="500px" />
</div>

## Paper link
- [Google Drive](https://drive.google.com/file/d/1zsUnEmXTLwXqTKytpv3dWTqEreK90_bI/view?usp=sharing)
## Citation
Please cite the above publication if you use the code or compare with the ASRCF tracker in your work. Bibtex entry:
```
@InProceedings{Dai_2019_CVPR,  
  author = {Dai, Kenan and Wang, Dong and Lu, Huchuan and Sun, Chong and Li, Jianhua},  
  title = {Visual Tracking via Adaptive Spatially-Regularized Correlation Filters},  	
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  	
  month = {June},  
  year = {2019}  
}  
```
## Installation
1. Clone the GIT repository:
```
 $ git clone https://github.com/Daikenan/ASRCF.git
```
2. Clone the submodules.  
   In the repository directory, run the commands:
```
   $ git submodule init  
   $ git submodule update
```
3. Start Matlab and navigate to the repository.  
   Run the install script:
```
   |>> install
```
4. Run the demo script to test the tracker:
```
   |>> demo_ASRCF
```   
## Use different GPU cards
- We use GPU card 1 by default, if you want to use other GPU cards, such as card 2, you can run these code in matlab Command Window.
```
　opts.gpus=[2];
　prepareGPUs2(opts,ture);
```
## Spatial regularization
- In our demo, we show the spatial adaptive regularization by default, but this is time consuming.
if you want to close it, you need just set `params.show_regularization = 0` in `run_ASRCF.m`.
