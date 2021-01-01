import tensorflow as tf
import time
import sys
def get_data_string(steps,current_step,loss):
   total_bar = 50
   increase_step = int(steps/total_bar + 0.5)
   for i in range(1,83+1):
      if i%increase_step == 0:
        dashes = int(i/increase_step + 0.5)
        dots   = total_bar - dashes
        string = '='*( dashes - 1)
        if not i == steps and not i == 0:
          string += '>'
        string += '.'*(dots - 1)

   data_String = '{0}/{1} :'.format(current_step,steps) + string
   data_String += ' Loss: ' + str(loss)
   return data_String

@tf.function
def train_one_step(model,inp, y_true,loss_func,optimizer,apply_regularization = False):

    with tf.GradientTape() as tape:
        y_pred = model(inp)
        loss   = loss_func(y_true, y_pred)
        if apply_regularization:
            loss += tf.math.sum(model.losses)
        grads  = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

    return loss,model


def train(train_dataset, val_dataset, epochs, steps_per_epoch, val_steps, checkpoint, checkpoint_prefix, model, loss_func, optimizer, save_after = None,apply_regularization = False):
    epoch = checkpoint.epoch.numpy()

    while epoch < epochs:
        step  = 0
        avg_loss = 0.
        print('Epoch :' + str(epoch))
        for img,mask,score in train_dataset.take(steps_per_epoch):
            y_true = {'mask':mask, 'score':score}
            loss,model = train_one_step(model,img,y_true,loss_func,optimizer,apply_regularization)
            loss       = float(loss)
            avg_loss   = (avg_loss*step + loss)/(step + 1)

            string = get_data_string(steps_per_epoch, step, avg_loss)
            sys.stdout.write('\r' + string)
            time.sleep(0.01)
            step += 1
        step = 0
        val_avg_loss = 0.
        for img, mask, score in val_dataset.take(val_steps):
            y_true = {'mask': mask, 'score': score}
            y_pred = model(img)
            loss   = loss_func(y_true,y_pred)
            val_avg_loss = (val_avg_loss * step + loss) / (step + 1)
            step += 1
        string += ' Val Loss: ' + str(val_avg_loss)
        sys.stdout.write('\r' + string)
        print()
        tf.summary.scalar('Loss',avg_loss)
        tf.summary.scalar('Val_Loss',val_avg_loss)
        if save_after is not None:
            if epoch%save_after == 0:
                checkpoint.save(checkpoint_prefix)

    return model




