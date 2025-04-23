# Deception detection (Pytorch)

## Flowchart of deception detection

The overall flowchart for deception detection is illustrated below. We combine many features into a vector and then apply SVM to classification.

<p align="center">
 <img src="https://github.com/come880412/Deception_detection/blob/main/img/Flowchart%20.png" width=50% height=50%>
</p>

## Face alignment

Because the face can be many angles, we need to align the face before using it.

<p align="center">
 <img src="https://github.com/come880412/Deception_detection/blob/main/img/face%20alignment.png" width=50% height=50%>
</p>

## User instrctions

Our deception detection system comprises four parts：

1. 3D landmarks displacement
2. Emotion Unit
3. Action Unit

### Install Packages

Please see the `requirements.txt` for more details.

## Inference

```python=
python lie_GUI.py
```
