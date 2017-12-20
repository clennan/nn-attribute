# Layer-wise Relevance Propagation (LRP) to attribute importance of input pixels
Generates heatmaps that highlights image pixels that were important for a CNN classification result.
Currently only supports VGG16. The CNN and relevance propagation rules are implemented in TensorFlow.

Example:

<div align="center">
  <img src="https://raw.githubusercontent.com/clennan/nn-attribute/master/src/readme_files/goose.jpg" style="width: 200px;">
  <img src="https://raw.githubusercontent.com/clennan/nn-attribute/master/src/readme_files/goose_hm.png" style="width: 200px;"><br><br>
</div>

For more details on LRP see:  
http://heatmapping.org  
http://journals.plos.org/plosone/article?id=10.1371/journal.pone.  
http://www.sciencedirect.com/science/article/pii/S0031320316303582?via%3Dihub  
https://arxiv.org/abs/1512.02479


**Setup:**
- create Python 3 virtual environment and activate the environment
```
virtualenv -p python3 ~/nn-attribute
source ~/nn-attribute/bin/activate
```

- install Python dependencies with
```
pip3 install -r requirements.txt
```


- download VGG16 pretrained weights
```
wget -P data/vgg16 ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy
```


**Generate heatmaps:**
- choose image file to generate heatmaps for (e.g. ./data/goose.jpg)
- choose alpha (recommendation: alpha=2)
```
python -m heatmap \
--image-file data/goose.jpg \
--target-file heatmap.pdf \
--model vgg16 \
--weights-file data/vgg16.npy \
--classes-file data/vgg16/classes.json \
--alpha 2
```

- generated heatmaps are saved in `results` folder
