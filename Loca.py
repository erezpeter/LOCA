# See E. Peterfreund, O. Lindenbaum, F. Dietrich, T. Bertalan, M. Gavish, I.G. Kevrekidis and R.R. Coifman,
# "LOCA: LOcal Conformal Autoencoder for standardized data coordinates",
# https://www.pnas.org/doi/full/10.1073/pnas.2014627117
#
#
# -----------------------------------------------------------------------------
# Author: Erez Peterfreund , Ofir Lindenbaum
#         erezpeter@gmail.com  , ofir.lindenbaum@yale.edu , 2020
# 
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------



import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# Validates whether all the values of x are in the range of [minVal,maxVal]
# x- should be a numpy array
def val_num(x,minVal= None,maxVal=None, errMsg =None):
    if (minVal is not None):
        if x<minVal:
            if errMsg is None:
                raise Exception('The value '+str(x)+' should be bigger then '+str(minVal))
            raise Exception(errMsg+ ' Actual val: '+str(x))
    
    if not (maxVal is None):
        if x>maxVal:
            if errMsg is None:
                raise Exception('The value '+str(x)+' should be below '+str(maxVal))
            raise Exception(errMsg+ ' Actual val: '+str(x))


# Calculates the covariance of the data
# x- N x M x d tensorflow tensor, where N is the amount of clouds/short-bursts, M is the 
#                      amount of samples in a cloud/short-burst and d is the coordinates
def tf_cov(x):
    val_num(len(np.shape(x)),minVal=3, maxVal=3,errMsg='The data should be a 3-d tensor.')
            
    x_no_bias= x- tf.reduce_mean(x,axis=1,keepdims=True) 
        
    cov_x= tf.matmul(tf.transpose(x_no_bias ,[0,2,1]), x_no_bias)/tf.cast(tf.shape(x)[1]-1, tf.float32)
    return cov_x


def get_activation_layer(inputTensor, currActivation):
    if currActivation == 'relu':
        outputTensor = tf.nn.relu(inputTensor)
    elif currActivation == 'l_relu':
        outputTensor = tf.nn.leaky_relu(inputTensor)
    elif currActivation == 'sigmoid':
        outputTensor = tf.nn.sigmoid(inputTensor)
    elif currActivation == 'tanh':
        outputTensor = tf.nn.tanh(inputTensor)
    elif currActivation == 'none':
        outputTensor =inputTensor
    else:
        raise Exception('Error: '+str(model.act_type_dec)+' is not supported. The activations that are supported- relu,l_relu,tanh,sigmoid,none')

    return outputTensor


# The function returns the output of the net and its weights that are defined by the input args. The final layer of the 
# network include only an affine transformation and no activation function.
# INPUT:
#         input_tensor- a tensor where its last dimension includes the coordinates of the different samples.
#         input_dim- a positive integer. The dimension of the data.
#         layers- a list that includes the amount of neurons in each layer
#         act_type- a string defining the activation function (see get_activation_layer())
#         amount_layers_created - an integer. The amount of layers created so far for the net.
#
# OUTPUT:
#         layer_out - a 3 dimensional tensor (? x ? x output dimension) of type tensorflow variable. 
#                     This variable represents the final layer of the generated neural network
#         nnweights - A list that contains the weights and biases variables that were used in this network
#
def generateCoder(input_tensor, input_dim,layers,act_type,amount_layers_created=0):
    
    layer_out= input_tensor
    prev_node_size= input_dim
    nnweights =[]
    
    
    for i in range(len(layers)):
        layer_name = 'layer' + str(amount_layers_created+i+1)
        
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable('weights', [prev_node_size, layers[i]], \
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            
            biases = tf.get_variable('biases', [layers[i]], \
                                     initializer=tf.constant_initializer(0))
            
            nnweights.append(weights)
            nnweights.append(biases)

            layer_out = (tf.tensordot(layer_out, weights,axes=[[-1],[0]]) + biases) # Softmax                
                
            # The activation layer will not be used on the final layer                
            if i<len(layers)-1:
                layer_out= get_activation_layer(layer_out, act_type)

            
            prev_node_size = layers[i]
            
    return layer_out, nnweights

    

# The function gets a Loca class object and generates a neural network that is saved in the net fields
def generateNeuralNet(net):
   
    net.LR = tf.placeholder(tf.float32, shape=(), name="init")

    net.X = tf.placeholder(tf.float32, [None, None, net.input_dim]) 
    
    net.embedding, net.nnweights_enc=generateCoder(net.X,net.input_dim, net.encoder_layers,net.act_type,\
                                           amount_layers_created=0)
    
    net.reconstruction, net.nnweights_dec=generateCoder(net.embedding,  net.embedding_dim, net.decoder_layers,\
                                       net.act_type_dec, amount_layers_created=len(net.encoder_layers)+1)
                

        
# The Tensorflow implementation of LOCA
class Loca(object): 
    # The init function.
    # INPUT:
    #             - clouds_var - float32. The assumed variance of each cloud ( see sigma squared in the paper).
    #             - encoder_layers - A list that includes the amount of neurons in each layer of the generated encoder. 
    #             - decoder_layers - A list that includes the amount of neurons in each layer of the generated decoder.
    #             - activation_enc - 'relu'/'l_relu','sigmoid','tanh','none' - the activation function that will be used 
    #                                 in the encoder
    #             - activation_dec(Optional) - 'relu'/'l_relu','sigmoid','tanh','none' - the activation function that 
    #                                 will be used in the decoder. If not supplied activation_enc will be used.
    def __init__(self, clouds_var, encoder_layers,decoder_layers,activation_enc, activation_dec=None ):
        
        # Neural net params
        self.encoder_layers= encoder_layers
        self.decoder_layers= decoder_layers
        
        self.input_dim = self.encoder_layers[0]
        self.embedding_dim = self.encoder_layers[-1]
        self.output_dim= self.decoder_layers[-1]
        
        if self.decoder_layers[-1] != self.input_dim:
            raise Exception('The final layer of the decoder should have the same dimension as the input')
        
        self.act_type = activation_enc
        if activation_dec is not None:
            self.act_type_dec = activation_dec
        else:
            self.act_type_dec = self.act_type
        
        
        self.nnweights_enc = []
        self.nnweights_dec = []
        
        
        # Loss related params
        self.burst_var= clouds_var

                
        # Training history
        self.epochs_done=0
        self.lr_training=[]
        
        self.best_weights=None # Determined by the validation loss
        self.best_loss= np.inf
        self.best_rec= np.inf
        self.best_white= np.inf
        
        
        # Generate neural network
        G = tf.Graph()
        with G.as_default():
            self.sess = tf.Session(graph=G)
            generateNeuralNet(self)
            
            normalized_cov_code= tf_cov(self.embedding)/(self.burst_var)-\
                        tf.expand_dims(tf.eye(self.embedding_dim),axis=0)
            
            self.white_loss=tf.reduce_sum(tf.reduce_mean(normalized_cov_code**2, axis=0))
                   
            self.rec_loss=tf.reduce_mean(tf.reduce_sum((self.reconstruction-self.X)**2,axis=-1))

            
            self.train_step_whitening = tf.train.AdamOptimizer(self.LR).minimize(self.white_loss,var_list=self.nnweights_enc)
                
            self.train_step_recon = tf.train.AdamOptimizer(self.LR).minimize(self.rec_loss,\
                                         var_list=self.nnweights_dec+self.nnweights_enc)
            
            
            self.train_step_recon_dec = tf.train.AdamOptimizer(self.LR).minimize(self.rec_loss,var_list=self.nnweights_dec)


            init_op = tf.global_variables_initializer()

            self.saver = tf.train.Saver()
            
        
        self.sess.run(init_op)
       
   
    
    # The function trains the neural net on data_train. If evaluate_every is given the function evaluates the neural net's
    # performance on data_train and data_val(if given). If data_val and evaluate_every is given the function will restore 
    # the best version of the neural network based before returning. The best version means the neural network with the lowest
    # loss on data_val throughout the evluations steps.
    #                        
    # data_train- a Nx M x d tensor of data, where N indicates the amount of clouds/short bursts, 
    #                                  M indicates the amount of points in each cloud/ shorts burst
    #                                  and d indicates the dimension of the each sample
    # amount_epochs-   int. The amount of epochs to run.
    # lr-          float32. The learning that will be used throughout the training
    # batch_size - int. The batch size that will be used in the GD.
    # data_val(Optional)-    Same as in train_data but for validation (the values of m and d of this
    #                        tensor should be the same as in train_data). 
    # evaluate_every(Optional) - int. The amount of epochs that will be passed between the evaluation of the losses 
    #                                 based on the training data (data_train) and validation data (data_val) if is given.
    # verbose(Optional):          Boolean - Enables the printing of the losses evaluated evalutate_every epochs.
    # train_only_decoder(Optional):    Boolean. If True the training will only apply optimize the reconstruction loss, 
    #                                  and will update only the weights in the decoder.
    def train(self, data_train,amount_epochs,lr=0.1, batch_size=None, data_val=None, evaluate_every=None,verbose=False,
              train_only_decoder= False):
                            
        N_train= data_train.shape[0]
        N_val=0
        
        if data_val is not None:
            if data_val.shape[0]>0:
                N_val= data_val.shape[0]
               
        
        if batch_size is None:
            batch_size= N_train
        
        val_num(len(np.shape(data_train)), minVal=3, maxVal=3,errMsg='data_train should be a 3d tensor')
        val_num(batch_size,minVal=1, errMsg='btach_size should be at least 1')
                   
            
        for epoch in range(amount_epochs):

            for i in range(0,N_train,batch_size):
                
                if i+batch_size<=N_train:
                    indexes= np.arange(i,i+batch_size)
                else:
                    indexes= np.mod(np.arange(i,i+batch_size),N_train)
               
                batch_xs=data_train[indexes ,:,:]
                                    
                if not train_only_decoder:
                    if epoch% 2:
                        _ = self.sess.run([self.train_step_whitening], feed_dict={self.X: batch_xs, self.LR: lr})
                    else:
                        _ = self.sess.run([self.train_step_recon], feed_dict={self.X: batch_xs, self.LR: lr})
                        
                        
                else: # train only the decoder using the reconstruction loss
                    _ = self.sess.run([self.train_step_recon_dec], feed_dict={self.X: batch_xs, self.LR: lr})
                    
            
            
            
            # Evaluation stage
            if evaluate_every is not None:
                if (epoch+1) % evaluate_every == 0 and (verbose or early_stopping):

                    overall_train_white_loss= 0.
                    overall_train_rec_loss= 0.
                    # Train
                    for i in range(0,N_train,batch_size):
                        max_ind= np.min([N_train, i+ batch_size])
                        batch_xs=data_train[i:max_ind ,:,:]
                        
                        rec_loss_train, white_loss_train= self.sess.run([ self.rec_loss, self.white_loss], feed_dict={\
                                        self.X: batch_xs})

                        overall_train_white_loss+= white_loss_train*(max_ind-i)/N_train
                        overall_train_rec_loss+= rec_loss_train*(max_ind-i)/N_train


                    if data_val is not None:
                        # Validation 
                        overall_val_white_loss=0. 
                        overall_val_rec_loss =0.
                        for i in range(0,N_val,batch_size):
                            max_ind= np.min([N_val, i+ batch_size])
                            batch_xs=data_val[i:max_ind ,:,:]

                            rec_loss_val, white_loss_val= self.sess.run([ self.rec_loss, self.white_loss], feed_dict={\
                                        self.X: batch_xs})

                            overall_val_white_loss+= white_loss_val*(max_ind-i)/N_val
                            overall_val_rec_loss+= rec_loss_val*(max_ind-i)/N_val


                    if verbose:
                        if data_val is not None:
                            print("Epoch:", '%04d' % (self.epochs_done+1), "Train : white=", "{:.5f}".format(overall_train_white_loss),\
                              "rec={:.5f}".format(overall_train_rec_loss), "     Val: : white=", "{:.5f}".format(overall_val_white_loss),\
                              "rec={:.5f}".format(overall_val_rec_loss))
                        else:
                            print("Epoch:", '%04d' % (self.epochs_done+1), "Train : white=", "{:.5f}".format(overall_train_white_loss),\
                              "rec={:.5f}".format(overall_train_rec_loss))

                    
                    # Saves the best version of the neural network based on its validation loss
                    if data_val is not None:
                        curr_white_loss= overall_val_white_loss
                        curr_recon_loss= overall_val_rec_loss

                        if (self.best_loss> curr_recon_loss + curr_white_loss):

                                self.best_weights= self.get_current_weights()
                                self.best_loss= rec_loss_val + white_loss_val
                                self.best_rec= curr_recon_loss
                                self.best_white= curr_white_loss
                                

            self.epochs_done+=1
        
        self.load_weights_lowest_val()
    
    
    
    
    def eval_whitening_loss(data,batch_size=100):
        val_num(len(np.shape(data_train)), minVal=3, maxVal=3,errMsg='data should be a 3d tensor')

        N= np.shape(data)
        overall_white_loss= 0.
            
        for i in range(0,N,batch_size):
            max_ind= np.min([N, i+ batch_size])
            batch_xs=data[i:max_ind ,:,:]
            white_loss= self.sess.run(self.white_loss, feed_dict={self.X: batch_xs})

            overall_white_loss+= white_loss*(max_ind-i)/N
        return overall_train_white_loss
        
        
    def eval_recon_loss(data,batch_size=100):
        val_num(len(np.shape(data_train)), minVal=3, maxVal=3,errMsg='data should be a 3d tensor')

        N= np.shape(data)
        overall_recon_loss= 0.
            
        for i in range(0,N,batch_size):
            max_ind= np.min([N, i+ batch_size])
            batch_xs=data[i:max_ind ,:,:]
            recon_loss= self.sess.run(self.rec_loss, feed_dict={self.X: batch_xs})

            overall_recon_loss+= recon_loss*(max_ind-i)/N
        return overall_train_white_loss
        
    
    
    # The function inputs the data into the neural network and returns the embedding and reconstruction of the data.
    # INPUT:
    #         data-  a 2 or 3 dimensional tensor, where its last dimension indicates the coordinates of the data.
    # 
    # OUTPUT:
    #         embedding - same structure as data. Includes the embedding of the different input datapoints.
    #         reconstruction - same structure as data. Includes the reconstruction of the different input datapoints.
    def test(self,data):
        val_num(len(np.shape(data)),minVal= 2,maxVal=3, errMsg='The data should be a 2d or 3d tensor.')
                            
        new_data= data +0.
        if len(np.shape(data))==2:
            new_data= np.expand_dims(new_data,axis=1)
            
        embedding ,reconstruction = self.sess.run([self.embedding,self.reconstruction], feed_dict={self.X: new_data})
        
        if len(np.shape(data))==2:
            return embedding[:,0,:], reconstruction[:,0,:]
            
        return embedding,reconstruction   
    
    # The function inputs the data into the neural network and returns the embedding and reconstruction of the data.
    # INPUT:
    #         data-  a 2/3 dimensional tensor, where its last dimension indicates the coordinates of the data. It will include
    #                the input for the embedding layer of the neural network.
    # 
    # OUTPUT:
    #         newData- me structure as data. Includes the reconstruction of the different input datapoints.
    def decode(self,data):
        val_num(len(np.shape(data)),minVal= 2,maxVal=3, errMsg='The data should be a 2d or 3d tensor.')
        decoder_weights = self.get_current_weights()[1]
        
        newData = data+0.
        if len(np.shape(data))==2:
            newData=np.expand_dims(newData,axis=1)
                
        for i in range(0,len(decoder_weights),2):
            newData= np.matmul(newData,decoder_weights[i])+decoder_weights[i+1]
            
            
            # The last layer of the decoder is linear
            if i< len(decoder_weights)-2:
                if self.act_type_dec=='relu':
                    newData= np.maximum(newData,0)
                    
                elif self.act_type_dec=='l_relu':
                    alpha=0.2
                    newData= np.maximum(newData,0) + alpha*np.minimum(newData,0)
                elif self.act_type_dec=='tanh':
                    newData= np.tanh(newData)
                elif self.act_type_dec=='sigmoid':
                    newData = 1/(1 + np.exp(-newData)) 
                elif self.act_type_dec=='none':
                    newData= newData
                else:
                    raise Exception('Error: '+str(self.act_type_dec)+\
                                    ' is not supported. The activations that are supported- relu,l_relu,tanh,sigmoid,none')
                
                    
        if len(np.shape(data))==2:
            newData= newData[:,0,:]
        return newData
    
    # The method updates the neural network weights with the weights of the neural network that achieved 
    # the lowest validation loss throughout training
    def load_weights_lowest_val(self):
         self.load_weights( self.best_weights[0], self.best_weights[1])

    # The function returns the weights of the current neural network as two lists.
    # OUTPUT: 
    #        we - A list that contains the neural network weights of the encdoer.
    #        wd - A list that contains the neural network weights of the decoder.
    def get_current_weights(self):
        we = self.sess.run(self.nnweights_enc)
        wd= self.sess.run(self.nnweights_dec)
        return we, wd
   
    # The function loads the given weights into the neural network.
    # INPUT:
    #         encoderWeights - A list containing the values of the encoder part of the neural network.
    #         decoderWeights - A list containing the values of the decoder part of the neural network.
    def load_weights(self,encoderWeights, decoderWeights):
        for i in range(len(encoderWeights)):
            self.sess.run(self.nnweights_enc[i].assign(encoderWeights[i]))
            
        for i in range(len(decoderWeights)):
            self.sess.run(self.nnweights_dec[i].assign(decoderWeights[i]))
                            
