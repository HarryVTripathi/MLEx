import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

EMBEDDING_SIZE = 50
n_words = 0
MAX_LABEL = 15
WORDS_FEATURE = 'words'
TRAIN_DATA = 500
TEST_DATA = 50

dbpedia = tf.contrib.learn.datasets.load_dataset(
  'dbpedia', size='small', test_with_fake_data=False
)

# shuffling the training and test data 
np.random.seed(10)

shuffle_indices = np.random.permutation(np.arange(len(dbpedia.train.data)))
data_shuffled = dbpedia.train.data[shuffle_indices]
target_shuffled = dbpedia.train.target[shuffle_indices]

train_data = data_shuffled[:TRAIN_DATA]
train_target = target_shuffled[:TRAIN_DATA]

shuffle_indices = np.random.permutation(np.arange(len(dbpedia.test.data)))
data_shuffled = dbpedia.test.data[shuffle_indices]
target_shuffled = dbpedia.test.target[shuffle_indices]

test_data = data_shuffled[:TEST_DATA]
test_target = target_shuffled[:TEST_DATA]

print(np.unique(dbpedia.train.target))

x_train = pd.Series(train_data[:, 1])
y_train = pd.Series(train_target)

x_test = pd.Series(test_data[:, 1])
y_test = pd.Series(test_target)

# find the max length of paragraph across testing and train datasets
# to represent every document as a tensor

# We want every document to have exactly the same no. of words
max_document_length_train = max([len(x.split(" ")) for x in x_train])
max_document_length_test = max([len(x.split(" ")) for x in x_test])

MAX_DOC_LENGTH = max(max_document_length_test, max_document_length_train)

# longer documents will be truncated, and shorter ones will be padded

# converting document to a numeric vector
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOC_LENGTH)
n_words = vocab_processor.vocabulary_

x_train = np.array(list(vocab_processor.fit_transform(x_train)))
x_test = np.array(list(vocab_processor.fit_transform(x_test)))

print(x_train[:3])

tf.reset_default_graph()


def estimator_spec_for_softmax_classification(logits, labels, mode):
  # logits: output layer to which softmax is to be applied
  # labels: actual labels which we use to calculate the loss
  # mode: how we want to run our model

  # logits layer outputs a set of probablity
  # the label with the highest probability is the predicted label output
  predicted_classes = tf.argmax(logits, 1)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      prediction={
        'class': predicted_classes,
        'prob': tf.nn.softmax(logits)
      }
    )
  
  onehot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0)

  # adds softmax layer to the logits layer
  loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels,
    logits=logits
  )

  # Training mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
  
  eval_metric_ops = {
    'accuracy': tf.metrics.accuracy(
      labels=labels, predictions=predicted_classes
    )
  }

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    eval_metric_ops=eval_metric_ops
  )

def rnn_model(features, labels, mode):
  
  # Find the word-vector encodings for all the input instances that have been passed in
  word_vectors = tf.contrib.layers.embed_sequence(
    features[WORDS_FEATURE],
    vocab_size=n_words,
    embed_dim=EMBEDDING_SIZE
  )

  #  word_vectors = [batch_size, n_steps, EMBEDDING_SIZE]
  # print(word_vectors.shape)

  # word_list = [no_of_words, EMBEDDING_SIZE]
  word_list = tf.unstack(word_vectors, axis=1)
  # print(word_list.shape)

  cell = tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)

  # using static rnn
  _, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return estimator_spec_for_softmax_classification(
    logits=logits, labels=labels, mode=mode
  )

classifier = tf.estimator.Estimator(model_fn=rnn_model)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={WORDS_FEATURE: x_train},
  y=y_train,
  batch_size=len(x_train),
  num_epochs=None,
  shuffle=True
)

classifier.train(input_fn=train_input_fn, steps=10)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={WORDS_FEATURE: x_test},
  y=y_test,
  num_epochs=1,
  shuffle=False
)

predictions = classifier.predicted(input_fn=test_input_fn)


# Extracting classes of predicted data
y_predicted = np.array(list(p['class'] for p in predictions))

y_predicted = y_predicted.reshape(np.array(y_test).shape)
