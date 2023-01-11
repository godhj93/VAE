import tensorflow as tf
import pandas as pd
import pickle
import os 

class dataloader:
    '''
    Date: 2023.01.05
    Author: H. Shin
    Description: Dataloader for high speed collision avoidance
    Data structure: (image, next_image, imu, next_imu, dir_vector, next_dir_vector, action)
    '''
    def __init__(self, batch_size=32, csv_path='train_ds.csv'):
        
        self.batch_size = batch_size
        self.data_list = pd.read_csv(csv_path, header=None)
         
        self.filenames = self.data_list[0].tolist()
        # for item in :
        #     item = 'data/' + item
        #     self.filenames.append(item)
        

    def length(self):
        
        # print(f"Batch size: {self.batch_size}")
        # print(f"Steps per epoch: {len(self.filenames)//self.batch_size}")
        return len(self.filenames)//self.batch_size

    def get_image(self, filename):

        filename = bytes.decode(filename.numpy())
        with open(os.path.join('data',filename), 'rb') as f:
            data =  pickle.load(f)

        image = data['image']
        
        return image

    def get_next_image(self, filename):

        filename = bytes.decode(filename.numpy())
        with open(os.path.join('data',filename), 'rb') as f:
            data =  pickle.load(f)

        next_image = data['next_image']
        return next_image

    def get_next_image(self, filename):

        filename = bytes.decode(filename.numpy())
        with open(os.path.join('data',filename), 'rb') as f:
            data =  pickle.load(f)

        next_image = data['next_image']
        return next_image

    def get_imu(self, filename):

        filename = bytes.decode(filename.numpy())
        with open(os.path.join('data',filename), 'rb') as f:
            data =  pickle.load(f)

        imu = data['state']
        return imu

    def get_next_imu(self, filename):

        filename = bytes.decode(filename.numpy())
        with open(os.path.join('data',filename), 'rb') as f:
            data =  pickle.load(f)

        imu = data['next_state']
        return imu

    def get_dir_vec(self, filename):

        filename = bytes.decode(filename.numpy())
        with open(os.path.join('data',filename), 'rb') as f:
            data =  pickle.load(f)

        dir_vector = data['dir_vector']
        return dir_vector

    def get_next_dir_vec(self, filename):

        filename = bytes.decode(filename.numpy())
        with open(os.path.join('data',filename), 'rb') as f:
            data =  pickle.load(f)

        next_dir_vector = data['next_dir_vector']
        return next_dir_vector

    def get_action(self, filename):

        filename = bytes.decode(filename.numpy())
        with open(os.path.join('data',filename), 'rb') as f:
            data =  pickle.load(f)

        action = data['action']
        return action

    def _parse_function(self, filename):

        image = tf.py_function(self.get_image, inp=[filename], Tout=tf.float32)
        next_image = tf.py_function(self.get_next_image, inp=[filename], Tout=tf.float32)

        imu = tf.py_function(self.get_imu, inp=[filename], Tout=tf.float32)
        next_imu = tf.py_function(self.get_next_imu, inp=[filename], Tout=tf.float32)

        dir_vector = tf.py_function(self.get_dir_vec, inp=[filename], Tout=tf.float32)
        next_dir_vector = tf.py_function(self.get_next_dir_vec, inp=[filename], Tout=tf.float32)

        action = tf.py_function(self.get_action, inp=[filename], Tout=tf.float32)

        return (image, imu, dir_vector), action, (next_image, next_imu, next_dir_vector)

    def get_batched_dataset(self):

        self.ds = tf.data.Dataset.from_tensor_slices(self.filenames)
        self.ds = self.ds.shuffle(buffer_size=len(self.filenames), reshuffle_each_iteration=True)
        self.ds = self.ds.repeat()
        self.ds = self.ds.map(map_func = self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        self.ds = self.ds.batch(batch_size=self.batch_size)

        return self.ds

        
        
        