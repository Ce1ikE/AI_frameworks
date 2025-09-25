## NOTE
most of these models will not be available in this REPO
due to file size that is why implementing a `model store` from
which you can download the model is better this avoids cluttering the library
the model store functions are the ones from the uniface REPO `https://github.com/yakhyo` but tweeked slightly to allow all models to use these function.

### Face representation : ArcFace
```python
MODEL_URLS: Dict[ArcFaceWeights, str] = {
    ArcFaceWeights.W600K_MBF: 'https://huggingface.co/WePrompt/buffalo_sc/resolve/main/w600k_mbf.onnx?download=true',
    ArcFaceWeights.W600K_R50: 'https://huggingface.co/maze/faceX/resolve/main/w600k_r50.onnx?download=true',
}

MODEL_SHA256: Dict[ArcFaceWeights, str] = {
    ArcFaceWeights.W600K_MBF: '9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f',
    ArcFaceWeights.W600K_R50: '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43',
}
```

### Face detection : RetinaFace
```python
MODEL_URLS: Dict[RetinaFaceWeights, str] = {
    RetinaFaceWeights.MNET_025: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv1_0.25.onnx',
    RetinaFaceWeights.MNET_050: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv1_0.50.onnx',
    RetinaFaceWeights.MNET_V1:  'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv1.onnx',
    RetinaFaceWeights.MNET_V2:  'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv2.onnx',
    RetinaFaceWeights.RESNET18: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_r18.onnx',
    RetinaFaceWeights.RESNET34: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_r34.onnx'
}

MODEL_SHA256: Dict[RetinaFaceWeights, str] = {
    RetinaFaceWeights.MNET_025: 'b7a7acab55e104dce6f32cdfff929bd83946da5cd869b9e2e9bdffafd1b7e4a5',
    RetinaFaceWeights.MNET_050: 'd8977186f6037999af5b4113d42ba77a84a6ab0c996b17c713cc3d53b88bfc37',
    RetinaFaceWeights.MNET_V1:  '75c961aaf0aff03d13c074e9ec656e5510e174454dd4964a161aab4fe5f04153',
    RetinaFaceWeights.MNET_V2:  '3ca44c045651cabeed1193a1fae8946ad1f3a55da8fa74b341feab5a8319f757',
    RetinaFaceWeights.RESNET18: 'e8b5ddd7d2c3c8f7c942f9f10cec09d8e319f78f09725d3f709631de34fb649d',
    RetinaFaceWeights.RESNET34: 'bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630'
}
```

### Face detection : YuNet
```python
MODEL_URLS: Dict[YuNetWeights, str] = {
    YuNetWeights.YUNET: 'https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx?download=true',
}

MODEL_SHA256: Dict[YuNetWeights, str] = {
    YuNetWeights.YUNET: '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
}
```

### Face detection : Viola Jones
```python
HAARCASCADE_URLS = {
    CascadeType.FRONTALFACE_DEFAULT: 'https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/data/haarcascades/haarcascade_frontalface_default.xml',
    CascadeType.FRONTALFACE_ALT: 'https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/data/haarcascades/haarcascade_frontalface_alt.xml',
    CascadeType.FULLBODY: 'https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/data/haarcascades/haarcascade_fullbody.xml',
}

HAARCASCADE_SHA256 = {
    CascadeType.FRONTALFACE_DEFAULT: '0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0',
    CascadeType.FRONTALFACE_ALT: '6281df13459cc218ff047d02b2ae3859b12ff14a93ffe8952f7b33fad7b9697b',  
    CascadeType.FULLBODY: '041745c71eef1b5c86aef224f17ce75b042d33314cc8f6757424f8bd8cd30aa1',     
}
```