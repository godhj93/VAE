from grpc import dynamic_ssl_server_credentials
import tensorflow as tf

class VAE(tf.keras.Model):

    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = Encoder(latent_dim=latent_dim)

        self.decoder = Decoder()

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, x, apply_sigmoid=False):
        
        logits = self.decoder(x)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class Encoder(tf.keras.Model):

    def __init__(self, latent_dim=32):
        super().__init__()

        self.latent_dim = latent_dim

        self.img_encoder = tf.keras.applications.DenseNet201(include_top=False, weights=None, input_shape=(128,128,8))
        
        self.imu_encoder = linear_encoder()

        self.action_encoder = linear_encoder(output_dim=16)

        self.dir_vec_encoder = linear_encoder()

        self.encoder = tf.keras.Sequential([

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.BatchNormalization(),


            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.BatchNormalization(),

            #No activation
            tf.keras.layers.Dense(units=self.latent_dim * 2)
        ])

    def call(self, x):

        s, action, s_next = x
        
        # Convert BCHW -> BHWC
        img = tf.transpose(s[0],perm=[0,2,3,1])
        imu = s[1]
        dir_vec = s[2]

        # Convert BCHW -> BHWC
        img_next = tf.transpose(s_next[0], perm=[0,2,3,1])
        imu_next = s_next[1]
        dir_vec_next = s_next[2]

        image_encoding = self.img_encoder(tf.concat([img, img_next],axis=-1))
        imu_encoding = self.imu_encoder(tf.concat([imu,imu_next],axis=2))
        action_encoding = self.action_encoder(action)
        dir_vec_encoding = self.dir_vec_encoder(tf.concat([dir_vec, dir_vec_next],axis=2))
        encodings = tf.concat([image_encoding, imu_encoding, action_encoding, dir_vec_encoding], axis=-1)

        return self.encoder(encodings)


class Decoder(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.img_decoder = img_decoder()

        self.imu_decoder = linear_decoder(output_dim=10)

        self.dir_vec_decoder = linear_decoder(output_dim=3)

    def call(self, x):

        (img, imu, dir_vec), a, c = x
        
        img_decode = self.img_decoder((img, a ,c))

        imu_decode = self.imu_decoder(tf.concat([tf.reshape(imu, shape=(-1,4*10)), a, c],axis=-1))

        dir_vec_decode = self.dir_vec_decoder(tf.concat([tf.reshape(dir_vec, shape=(-1,4*3)), a, c],axis=-1))
        
        return tf.transpose(img_decode, perm=[0,3,1,2]), imu_decode, dir_vec_decode

def linear_encoder(output_dim=4):
    '''Fully connected(linear) encoder'''
    return tf.keras.Sequential([
            tf.keras.layers.Dense(units=32),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(units=16),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(units=8),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(units=output_dim),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Reshape((4,4,1))
        ])

def linear_decoder(output_dim):
    '''Fully connected(linear) decoder'''
    return tf.keras.Sequential([

        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(4*output_dim),
        
        tf.keras.layers.Reshape((4,-1))
    ])

class img_decoder(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.cnn = tf.keras.Sequential([

                
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',
                    activation='relu'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',
                    activation='relu'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',
                    activation='relu'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',
                    activation='relu'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',
                    activation='relu'),
                tf.keras.layers.BatchNormalization(),
                # No activation
                tf.keras.layers.Conv2D(
                    filters=4, kernel_size=3, strides=1, padding='same'),
            ])

        self.reshape_a_and_c = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Reshape((8,8,1)),

            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', activation='relu'),
            ])

    def call(self, x):
        
        s_image, a, c = x

        s_image = tf.transpose(s_image, perm=[0,2,3,1])
        
        concat_a_c = self.reshape_a_and_c(tf.concat([a,c],axis=-1))
      
        return self.cnn(tf.concat([s_image, concat_a_c],axis=-1))
        