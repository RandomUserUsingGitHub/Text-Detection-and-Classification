import tensorflow as tf
from Generator.Utils.FileUtils import define_csv_name

# Define a function to parse a single row of the CSV file
def parse_csv_row(row):
    # Define column types and defaults to handle different data types
    column_types = [tf.string, tf.float32, tf.string]  # Assuming Bbox is in the format "x1, y1, x2, y2"
    column_defaults = ["", 0.0, ""]

    # Parse the CSV row
    columns = tf.io.decode_csv(row, record_defaults=column_defaults)
    
    # Split the Bbox string into a list of float values
    bbox = tf.strings.split([columns[1]], ',')
    bbox = tf.strings.to_number(bbox, out_type=tf.float32)
    
    # Create a dictionary mapping column names to their corresponding values
    features = {
        "Image_ID": columns[0],
        "Bbox": bbox,
        "Class_Label": columns[2]
    }
    
    return features

# Read the CSV file and create a dataset
csv_file = define_csv_name()
dataset = tf.data.TextLineDataset(csv_file).skip(1)  # Skip the header row
dataset = dataset.map(parse_csv_row)

# (Optional) Perform additional data preprocessing or transformations
# For example, you can decode image data and perform other data preprocessing here

# (Optional) Shuffle, batch, and prefetch the dataset
# dataset = dataset.shuffle(buffer_size=10000)
# dataset = dataset.batch(batch_size)
# dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Iterate through the dataset (for demonstration)
for data in dataset.take(5):  # Take the first 5 rows as an example
    print(data)

# Now you have a tf.data.Dataset ready to be used with TensorFlow for machine learning tasks
