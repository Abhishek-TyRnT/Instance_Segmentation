import tensorflow as tf
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K
class Joint_loss(Loss):

    def __init__(self, lambda_ = 1/32., h = 56., w = 56., names = 'Joint_Loss'):
        super(Joint_loss,self).__init__(name = names)
        self.lambda_  =  tf.constant(lambda_, name = 'lambda')
        self.h        =  tf.constant(h, name = 'height', dtype = tf.float32)
        self.w        =  tf.constant(w, name = 'width', dtype = tf.float32)


    def get_score_loss(self, y_true, y_pred):
        loss = -1*y_true*y_pred
        loss = K.sigmoid(loss)
        loss = -K.log(loss)
        return tf.cast(loss, tf.float32)


    def get_mask_loss(self, y_true, y_pred):
        loss = -1*y_true*y_pred
        loss =  K.sigmoid(loss)
        loss =  -K.log(loss)
        loss =  K.sum(loss, axis= [1, 2])
        return tf.cast(loss, tf.float32)


    def call(self, y_true, y_pred):
        true_mask, true_score = y_true['mask'], y_true['score']
        pred_mask, pred_score = y_pred['mask'], y_pred['score']

        mask_loss             = self.get_mask_loss(true_mask, pred_mask)
        score_loss            = self.get_score_loss(true_score, pred_score)

        index                 = tf.where(true_score == 1.0)
        mask_loss             = tf.gather(mask_loss, index)

        mask_loss             = mask_loss/(self.h*self.w)

        total_loss            = K.sum(mask_loss) + self.lambda_*score_loss
        return total_loss
