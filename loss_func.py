import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================
    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))
    A =
    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})
    B =
    ==========================================================================
    """

    #A:
    #u_o->outer/predicting word, v_c->center/context words
                                                                                                                                                                                                                                            
    #matrix_product = tf.mutiply(tf.transpose(true_w), inputs)
    matrix_product = tf.matmul(tf.transpose(true_w), inputs)
    exp_val = tf.exp(matrix_product)
    A = tf.log(exp_val)

    #B:
    #v_c->center/context word, \sum is sum reduced over columns
    #u_o is the output vector of the current output (label) word
    #u_w is the output vector of all label words in the batch
    
    #print ('A', A.shape)
    B = tf.log(tf.reduce_sum(exp_val, 1))
    #print ('B ', B.shape)
    
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weights: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    
    def custom_sigmoid(x):
        return -tf.log(1-(x*tf.log(2.718281828)) +log_correction)

    log_correction = 1e-10  #to avoid log(0) cases
    batch_size = inputs.shape[0].value
    embedding_size = inputs.shape[1].value
    k = len(sample)

    import numpy as np
    Pr = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    Pr_wo = tf.nn.embedding_lookup(Pr, labels)
    Pr_wo = tf.reshape(Pr_wo, [batch_size])
    Pr_wx = np.ndarray(shape=(k), dtype=np.float32)
    for i in range(0, k):
      Pr_wx[i] = unigram_prob[sample[i]]

    #FORMULA USED
    #s1 = SGM(ucT*uo + bo - log(k*Pr_wo))
    #s2 = SGM(ucT*ux + bx - log(k*Pr_wx))
    # - ( log(s1) + SUM(log(1-s2)) )
    #bo = bias vector wrt uo

    uo = tf.nn.embedding_lookup(weights, labels)
    uo = tf.reshape(uo, [-1, embedding_size])
    ux = tf.nn.embedding_lookup(weights, sample)
    matrix_s1 = tf.diag_part(tf.matmul(inputs, tf.transpose(uo)))
    matrix_s2 = tf.matmul(inputs, tf.transpose(ux))
    
    bo = tf.nn.embedding_lookup(biases, labels)
    tf.reshape(bo, [batch_size])
    bx = tf.nn.embedding_lookup(biases, sample)
    
    s1 = tf.sigmoid(matrix_s1 + bo - tf.log(k*Pr_wo+log_correction))
    #import pdb; pdb.set_trace()
    s2 = tf.sigmoid(matrix_s2 + bx - tf.log(k*Pr_wx+log_correction))
    return - (tf.log(s1+log_correction) + tf.reduce_sum(tf.log(1-s2+log_correction), 1))



    '''backup
    batch_size = inputs.shape[0].value
    embedding_size = inputs.shape[1].value

    k = len(sample)
    p=tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    p_w=tf.nn.embedding_lookup(p, labels)
    p_w=tf.reshape(p_w, [batch_size])
    ll=len(sample)
    p_x = np.ndarray(shape=(ll), dtype=np.float32)
    for i in range(0,ll):
      p_x[i]=unigram_prob[sample[i]]

    true_w = tf.nn.embedding_lookup(weights, labels)
    true_w = tf.reshape(true_w, [-1, embedding_size])
    sample_w = tf.nn.embedding_lookup(weights, sample)
    matrix_s1 = tf.diag_part(tf.matmul(inputs, tf.transpose(true_w)))
    #matrix_s2 = tf.matmul(tf.transpose(inputs), tf.transpose(sample_w))
    matrix_s2 = tf.matmul(inputs, tf.transpose(sample_w))
    print (k)
    
    bo = tf.nn.embedding_lookup(biases, labels)
    tf.reshape(bo, [batch_size])
    bx = tf.nn.embedding_lookup(biases, sample)
    #tf_softmax_correct = tf.placeholder("float", [batch_size,k])


    #1e-10
    log_correction = 1e-10
    #def tfsigmoid(x):
    #    return -tf.log(1-(x*tf.log(2.718281828)) +log_correction)

    
    s1 = tf.sigmoid(matrix_s1 + bo - tf.log(k*p_w+log_correction))
    #import pdb; pdb.set_trace()
    s2 = tf.sigmoid(matrix_s2 + bx - tf.log(k*p_x+log_correction))
    #return - (tf.log(s1+log_correction))
    return - (tf.log(s1+log_correction) + tf.reduce_sum(tf.log(1-s2+log_correction), 1))
    '''

