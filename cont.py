from keras.models import load_model
from data import data_iterator

batch_size = 32
epochs = 40
steps_per_epoch = 2000 / batch_size

model = load_model("model")

model.fit_generator(generator=data_iterator(batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs)

model.save("model")
