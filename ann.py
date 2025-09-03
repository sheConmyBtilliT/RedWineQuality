from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout


def get_ann_model(input_dim, output_dim):
    """
    Returns a compiled Artificial Neural Network model for multi-class classification.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
