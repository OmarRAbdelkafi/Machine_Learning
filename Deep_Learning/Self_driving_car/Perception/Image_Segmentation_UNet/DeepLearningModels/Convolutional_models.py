import tensorflow as tf
import DeepLearningModels.UNET.Unet_V0 as UnetV0
import matplotlib.pyplot as plt

def Set_unet_model_V0(img_height, img_width, num_channels):
    unet = UnetV0.unet_model_V0((img_height, img_width, num_channels))

    unet.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    unet.summary()

    return unet

def Fit_unet_model_V0(unet, processed_image_ds):
    #train model
    EPOCHS = 1
    VAL_SUBSPLITS = 5
    BUFFER_SIZE = 500
    BATCH_SIZE = 32

    processed_image_ds.batch(BATCH_SIZE)
    train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print(processed_image_ds.element_spec)

    model_history = unet.fit(train_dataset, epochs=EPOCHS)
    plt.plot(model_history.history["accuracy"])
    plt.show()

    return unet, train_dataset
