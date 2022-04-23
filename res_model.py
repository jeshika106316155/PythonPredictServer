import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, activations

def conv_bn_relu(input, k, ks, s, tail=False):
    c = layers.Conv2D(k, kernel_size=ks, strides=s, padding='same')(input)
    lnorm = layers.BatchNormalization()(c)
    if not tail:
        return activations.relu(lnorm)
    return lnorm

def res_w_bottleneck(input, k, downsample):
    nk = k * 4
    c1 = conv_bn_relu(input, k, 1, 1)
    if downsample:
        c2 = conv_bn_relu(c1, k, 3, 2)
        input = conv_bn_relu(input, nk, 2, 2)
    else:
        c2 = conv_bn_relu(c1, k, 3, 1)
        if input.shape[-1] != nk:
            input = conv_bn_relu(input, nk, 1, 1)
    c3 = conv_bn_relu(c2, nk, 1, 1, True)
    summation = layers.Add()([input, c3])
    return activations.relu(summation)

def seres_w_bottleneck(input, k, downsample=False, r=8):
    nk = k * 4
    c1 = conv_bn_relu(input, k, 1, 1)
    if downsample:
        c2 = conv_bn_relu(c1, k, 3, 2)
        input = conv_bn_relu(input, nk, 2, 2)
    else:
        c2 = conv_bn_relu(c1, k, 3, 1)
        if input.shape[-1] != nk:
            input = conv_bn_relu(input, nk, 1, 1)
    c3 = conv_bn_relu(c2, nk, 1, 1, True)
    bnorm = layers.BatchNormalization()(c3)
    squeeze = layers.GlobalAveragePooling2D()(bnorm)
    w1 = layers.Dense(k, activation='relu')(squeeze)
    w2 = layers.Dense(nk, activation='sigmoid')(w1)
    excite = layers.Multiply()([c3, w2])
    summation = layers.Add()([input, excite])
    return activations.relu(summation)


def res_block(input, k, lx, downsample=False, resfn=res_w_bottleneck):
    res = resfn(input, k, downsample=downsample)
    for _ in range(lx):
        res = resfn(res, k, downsample=False)
    return res

def pre_block(input, k, poolfn=layers.MaxPool2D):
    c = conv_bn_relu(input, k, 7, 2, False)
    return poolfn(pool_size=3, strides=2, padding='same')(c)

def mff(input, n_classes, lx, dropout_rate=0.3):
    dense = layers.Dense(input.shape[-1]*2, activation='relu')(input)
    assert dense.shape[-1] > lx * 8
    for _ in range(lx-1):
        dropout = layers.Dropout(dropout_rate)(dense)
        dense = layers.Dense(dense.shape[-1] // 8, activation='relu')(dropout)

    return layers.Dense(n_classes, activation='softmax')(dense)

def post_block(input, n_classes, lx, poolfn=layers.GlobalAveragePooling2D, dropout_rate=0.3):
    gp = poolfn()(input)
    return layers.Dense(n_classes, activation='softmax')(gp)

def resnet50(in_shape, n_classes, opt, units=[], loops=[], dropout_rate=0.3):
    in_layer = layers.Input(in_shape)
    print(in_layer.shape)
    res = pre_block(in_layer, 64)
    print(res.shape)
    res = res_block(res, units[0], loops[0])
    for idx, unit in enumerate(units[1:], 1):
        res = res_block(res, unit, loops[idx], True)
    
    out_layer = post_block(res, n_classes, loops[-1], layers.GlobalAveragePooling2D, dropout_rate)
    model = Model(in_layer, out_layer)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    model.summary()
    return model

def se_resnet50(in_shape, n_classes, opt, units=[], loops=[], dropout_rate=0.3):
    in_layer = layers.Input(in_shape)
    res = pre_block(in_layer, 64)
    res = res_block(res, units[0], loops[0], False, resfn=seres_w_bottleneck)
    for idx, unit in enumerate(units[1:], 1):
        res = res_block(res, unit, loops[idx], True, resfn=seres_w_bottleneck)
    
    out_layer = post_block(res, n_classes, loops[-1], layers.GlobalAveragePooling2D, dropout_rate)
    model = Model(in_layer, out_layer)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'categorical_accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    #model.summary()
    return model

def se_resnet50_gpc_in_chain_only(gpc_classifier, in_shape, n_classes, opt, units=[], loops=[], dropout_rate=0.3):
    gpc = se_resnet50(in_shape, n_classes, opt, units, loops, dropout_rate)
    gpc.load_weights(gpc_classifier)
    gpc.trainable = True    
    feature_extractor = Model(gpc.input, gpc.layers[-4].output)
    in_layer = layers.Input(in_shape)
    embedding = feature_extractor(in_layer)
    #conv = layers.Conv2D(1024, kernel_size=(3,3), strides=1, padding='same')(embedding)
    gp = layers.GlobalAveragePooling2D()(embedding)
    if dropout_rate > 0:
        gp = layers.Dropout(dropout_rate)(gp)
    dense1 = layers.Dense(128,activation='relu')(gp)
    dense2 = layers.Dense(2,activation='softmax')(dense1)
    model = Model(in_layer, dense2)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'categorical_accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    #model.summary()
    return model

if __name__ == '__main__':
    model = resnet50((200, 200, 3), 2, 'adam', [64, 128, 256, 512], [3, 4, 6, 3, 2], 0.3)