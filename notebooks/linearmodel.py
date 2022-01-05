from random import random
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']=''

class LinearModel():
    def __init__(self):
        self.slope=tf.Variable(0., name='slope',dtype=tf.float32)
        self.intercept= tf.Variable(0., name='intercept',dtype=tf.float32)
        self.mse = tf.keras.losses.MeanSquaredError()
        
    def loss(self,true,x):
        pred=self._predict(x)
        return self.mse(true,pred)
        
    def theta(self):
        return [self.slope.numpy(),self.intercept.numpy()]

    def settheta(self,theta):
        self.slope.assign(theta[0])
        self.intercept.assign(theta[1])
    
    def _loss_exact(self,x,y_true):
        return self.session.run(self._loss,feed_dict={self.x:x,self.y_true:y_true})

    
    def _predict(self,x):
        return x*self.slope+self.intercept
    
    def predict(self,x):
        return self._predict(x).numpy()

    
    def optimize(self,x,y_true,learning_rate,steps=20,batch_size=None):
        grad_v=[]
        path_v=[]

        self.opt=tf.keras.optimizers.SGD(learning_rate)
        
    
        for i in range(steps):
            if batch_size !=None:
                b_index=np.random.choice(range(len(x)),batch_size)
                batch=x[b_index]
                batch_y=y_true[b_index]
            else:
                batch=x
                batch_y=y_true
          
            with tf.GradientTape() as tape:
                _loss= self.loss(batch_y,batch)
            gradients = tape.gradient(_loss, [self.slope,self.intercept])
            path_v.append((self.slope.numpy(),self.intercept.numpy()))
            grad_v.append(gradients)
            self.opt.apply_gradients(zip(gradients,[self.slope,self.intercept] ))            
#           path_v.append((out[0][1],out[1][1]))

        return grad_v,path_v
