This is the code supplement for the paper:

E. Peterfreund, O. Lindenbaum, F. Dietrich, T. Bertalan, M. Gavish, I.G. Kevrekidis and R.R. Coifman,
"LOCA: LOcal Conformal Autoencoder for standardized data coordinates",
https://www.pnas.org/doi/full/10.1073/pnas.2014627117

-----------------------------------------------------------------------------
Author: Erez Peterfreund , Ofir Lindenbaum
        erezpeter@gmail.com , ofir.lindenbaum@yale.edu , 2020
        
        
Files: 
    - Loca.py - containing the implementation of the proposed neural network in Tensorflow.


Running LOCA:
1. Generate an instance of the Loca class by:

    model = Loca(clouds_var, encoder_layers,decoder_layers,activation_enc,activation_enc) 
    
    * INPUT:
    *
    * clouds_var - float32. The assumed variance of each cloud ( see sigma squared in the paper).
    * encoder_layers - A list that includes the amount of neurons in each layer of the generated encoder. 
    * decoder_layers - A list that includes the amount of neurons in each layer of the generated decoder.
    * activation_enc - 'relu'/'l_relu','sigmoid','tanh','none' - the activation function that will be used 
    *                                 in the encoder
    * activation_dec(Optional) - 'relu'/'l_relu','sigmoid','tanh','none' - the activation function that 
    *                                 will be used in the decoder. If not supplied activation_enc will be used.

2. Run training of the Loca instance (One can train the network multiple times sequnetially by calling the
   train function again and again)
   
   model.train(data_train, num_epoch,lr,batch_size,data_val, evaluate_every, verbose)
    
   * The function trains the neural net on data_train. If evaluate_every is given the function evaluates the neural net's
   * performance on data_train and data_val(if given). If data_val and evaluate_every is given the function will restore 
   * the best version of the neural network based before returning. The best version means the neural network with the lowest
   * loss on data_val throughout the evluations steps.
   *
   *
   * INPUT:
   *
   * data_train- a Nx M x d tensor of data, where N indicates the amount of clouds/short bursts, 
   *                                  M indicates the amount of points in each cloud/ shorts burst
   *                                  and d indicates the dimension of the each sample
   * num_epoch-   int. The amount of epochs to run.
   * lr-          float32. The learning that will be used throughout the training
   * batch_size - int. The batch size that will be used in the GD.
   * data_val(Optional)-    Same as in train_data but for validation (the values of m and d of this
   *                        tensor should be the same as in train_data). 
   * evaluate_every(Optional) - int. The amount of epochs that will be passed between the evaluation of the losses 
   *                                 based on the training data (data_train) and validation data (data_val) if is given.
   * verbose(Optional):          Boolean - Enables the printing of the losses evaluated evalutate_every epochs.
   * train_only_decoder(Optional):    Boolean. If True the training will only apply optimize the reconstruction loss, 
   *                                  and will update only the weights in the decoder.


3. Evaluate the neural network embedding and reconstruction on given data

    embedding, recontruction = model.test(y)
    
    * The function inputs the data into the neural network and returns the embedding and reconstruction of the data.
    * INPUT:
    *         data-  a 2/3 dimensional tensor, where its last dimension indicates the coordinates of the data.
    * 
    * OUTPUT:
    *         embedding - same structure as data. Includes the embedding of the different input datapoints.
    *         reconstruction - same structure as data. Includes the reconstruction of the different input datapoints.
