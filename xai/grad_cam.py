import numpy as np
import tensorflow as tf
import cv2

def generate_grad_cam(model, image, class_index):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(index=-3).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(image, axis=0))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0] 
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam.numpy(), (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    overlay = heatmap * 0.4 + np.float32(image)
    overlay = np.clip(overlay, 0, 1)

    return overlay
