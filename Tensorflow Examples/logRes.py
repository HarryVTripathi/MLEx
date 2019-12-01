import pandas as pd
import tensorflow as tf

TRAIN_DATA = './adult.data'
TEST_DATA = './adult.test'
MODEL_DIR = './linear_classifier'

CSV_COLUMNS = [
  "age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "gender",
  "captial_gain", "capital_loss", "hours_per_week", "native_country",
  "income_bracket"
]

df = pd.read_csv(
  TRAIN_DATA,
  names=CSV_COLUMNS,
  skipinitialspace=True,
  skiprows=1
)

# print(df.head())

TRIMMED_REORDERED_COLUMNS = [
  "age", "workclass", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "gender",
  "hours_per_week", "native_country", "income_bracket"
]

# df = df[TRIMMED_REORDERED_COLUMNS]

# print(df.head())

# explore all the methods of feature_column.
# set up categorical feature vector to feed these values into our estimator
gender = tf.feature_column.categorical_column_with_vocabulary_list(
  "gender", ["Female", "Male"]
)

# Get from df['race'].unique()
race = tf.feature_column.categorical_column_with_vocabulary_list(
  "race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
)

education = tf.feature_column.categorical_column_with_vocabulary_list(
  "education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
    'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
    '5th-6th', '10th', '1st-4th', 'Preschool', '12th']
)

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
  "marital_status", ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
    'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed']
)

relationship = tf.feature_column.categorical_column_with_vocabulary_list(
  "relationship", ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
    'Other-relative']
)

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
  "workclass", ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
    'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked']
)

age = tf.feature_column.numeric_column("age")

education_num = tf.feature_column.numeric_column("education_num")

hours_per_week = tf.feature_column.numeric_column("hours_per_week")

# Income may not vary at every age but it can greatly vary in age groups
age_buckets = tf.feature_column.bucketized_column(
  age, boundaries=[18, 25, 30, 40, 45, 50, 55, 60, 65]
)


# might keep changing for every census
occupation = tf.feature_column.categorical_column_with_hash_bucket(
  "occupation", hash_bucket_size=1000
)

native_country = tf.feature_column.categorical_column_with_hash_bucket(
  "native_country", hash_bucket_size=1000
)

base_columns = [
  gender, race, age, marital_status, occupation, workclass,
  native_country, age_buckets, education
]

crossed_columns = [
  tf.feature_column.crossed_column(
    ["education", "occupation"], hash_bucket_size=1000
  ),
  tf.feature_column.crossed_column(
    [age_buckets, "education", "occupation"], hash_bucket_size=1000
  ),
  tf.feature_column.crossed_column(
    ["native_country", "occupation"], hash_bucket_size=1000
  )
]

deep_columns = [
  education_num,
  hours_per_week
]

# defining the input function
def inputFn(fileName, num_epochs, shuffle):
  df = pd.read_csv(
    fileName,
    names=CSV_COLUMNS,
    skipinitialspace=True,
    skiprows=1
  )
  df = df[TRIMMED_REORDERED_COLUMNS]
  df = df.dropna(how="any", axis=0)

  # label = 1 if income>50K and 0, if income<50K
  labels = df["income_bracket"].apply(lambda x: ">50K" in x).astype(int)

  return tf.estimator.inputs.pandas_input_fn(
    x=df,
    y=labels,
    batch_size=100,
    num_epochs=num_epochs,
    shuffle=shuffle,
    num_threads=1
  )

linear_estimator = tf.estimator.LinearClassifier(
  model_dir=MODEL_DIR, feature_columns=base_columns + crossed_columns + deep_columns
)

linear_estimator.train(
  input_fn=inputFn(TRAIN_DATA, num_epochs=None, shuffle=True),
  steps=10
)

result = linear_estimator.evaluate(
  input_fn=inputFn(TRAIN_DATA, num_epochs=1, shuffle=False),
  steps=None
)

predictions = linear_estimator.predict(
  input_fn=inputFn(TRAIN_DATA, num_epochs=1, shuffle=False)
)

for keys in sorted(result):
  print("%s: %s" % (keys, result[keys]))

j = 0
for i, p in enumerate(predictions):
  if j>=10:
    break
  j=j+1
  print(i, p)

