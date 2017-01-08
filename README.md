# Darega (experimental)
Machine Learning Neural Network for identifying anime character faces (work-in-progress)

## System Requirements
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [Keras](https://github.com/fchollet/keras)

## Training

1. Install TensorFlow and Keras
2. Download training data: [animeface-character-dataset](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/) prepared by nagadomi. It contains 14490 faces of 177 anime characters.
3. Extract, rename folder "thumb" to "data", put it right next to the python scripts.
4. run `python train.py`
5. Observe the performance of the model, edit the training script, tune the parameters, change the model, and try again...

## Identifying character faces

run `python predict.py data\000_hatsune_miku\face_93_104_36.png`

It will use the pre-trained model `vocaloid.h5`, which is trained by using 5 vocaloid character faces: `000_hatsune_miku`, `018_kagamine_rin`, `033_kagamine_len`, `050_megurine_luka`, `099_kaito only`. It has only trained with 1000 samples in 972 epoch, with limited accuracy.

Sample output
```
Loading network...
Loading and preprocessing image...
Classifying image...
1/1 [==============================] - 0s
000_hatsune_miku
['86.94%', '0.83%', '0.50%', '0.00%', '4.51%']
```

## Notes
There are many hard-coded parameters in these scripts, make sure to edit and configure them correctly.