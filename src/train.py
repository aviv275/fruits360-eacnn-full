import argparse
import os
import tensorflow as tf  # noqa: F401
from tensorflow.keras import callbacks, optimizers
from .config import IMG_SIZE, BATCH_SZ, EPOCHS, LR, LOG_DIR, MODEL_DIR
from .dataload import get_datasets
from .eacnn import build_eacnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=BATCH_SZ)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()

    train_ds, val_ds, test_ds, class_names = get_datasets(img_size=IMG_SIZE, batch_size=args.batch)
    model = build_eacnn((*IMG_SIZE, 3), len(class_names))
    model.compile(
        optimizer=optimizers.Nadam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    cb = [
        callbacks.ModelCheckpoint(MODEL_DIR, save_best_only=True, save_format='tf'),
        callbacks.ReduceLROnPlateau(patience=5, verbose=1),
        callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        callbacks.TensorBoard(LOG_DIR)
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cb
    )
    # Save model diagram
    tf.keras.utils.plot_model(model, to_file=os.path.join(MODEL_DIR, 'eacnn_architecture.png'), show_shapes=True)
    # Always save the model at the end
    model.save(MODEL_DIR)

if __name__ == '__main__':
    main() 